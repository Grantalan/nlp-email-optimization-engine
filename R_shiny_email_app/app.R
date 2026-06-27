# ── Email Response Optimizer ──────────────────────────────────────────────────
# R Shiny app that:
#   1. Takes an email as input
#   2. Runs it through a logistic regression model (via Python/reticulate)
#      to predict the probability of a reply within 3 hours
#   3. Visualizes the email's features as a radar chart + gauge
#   4. Optionally generates an AI-revised email using RAG + OpenRouter

library(shiny)
library(bslib)
library(plotly)
library(reticulate)

# Point reticulate at the Anaconda Python that has sklearn, joblib, etc.
use_python("/opt/anaconda3/bin/python3", required = TRUE)

# Load the Python helpers — predict_helper.py runs the ML model,
# rag_helper.py runs the RAG + OpenRouter revision pipeline
source_python("predict_helper.py")
source_python("rag_helper.py")

# ── UI ────────────────────────────────────────────────────────────────────────

ui <- page_fillable(

  # Dark data-science dashboard theme
  theme = bs_theme(
    bootswatch = "darkly",
    base_font  = font_google("IBM Plex Mono")
  ),

  # Extra CSS for polish
  tags$head(tags$style(HTML("
    body           { background: #0f1117; color: #e2e8f0; }
    .navbar        { display: none; }
    .app-title     { font-size: 1.4rem; font-weight: 700; letter-spacing: 0.05em; color: #38bdf8; }
    .app-sub       { font-size: 0.8rem; color: #64748b; margin-top: 2px; }
    .card          { background: #1e2130; border: 1px solid #2d3148; border-radius: 12px; }
    .card-header   { background: transparent; border-bottom: 1px solid #2d3148;
                     font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
                     text-transform: uppercase; color: #64748b; }
    .prob-number   { font-size: 4rem; font-weight: 800; line-height: 1; font-family: 'IBM Plex Mono', monospace; }
    .prob-high     { color: #34d399; }
    .prob-mod      { color: #fbbf24; }
    .prob-low      { color: #f87171; }
    .badge-high    { background: #064e3b; color: #34d399; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
    .badge-mod     { background: #451a03; color: #fbbf24; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
    .badge-low     { background: #450a0a; color: #f87171; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
    .tip           { font-size: 0.82rem; color: #94a3b8; margin-top: 10px; border-left: 3px solid #38bdf8; padding-left: 10px; }
    .revised-box   { background: #0d1f0d; border: 1px solid #14532d; border-radius: 10px;
                     padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.75;
                     white-space: pre-wrap; color: #d1fae5; font-family: 'IBM Plex Mono', monospace; }
    .change-line   { font-size: 0.85rem; color: #94a3b8; padding: 5px 0;
                     border-bottom: 1px solid #2d3148; }
    .change-line:last-child { border-bottom: none; }
    .change-arrow  { color: #38bdf8; font-weight: 700; margin-right: 6px; }
    .rag-note      { font-size: 0.72rem; color: #475569; margin-top: 8px; }
    textarea.form-control { background: #0f1117 !important; color: #e2e8f0 !important;
                            border: 1px solid #2d3148 !important; border-radius: 8px;
                            font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
    .btn-primary   { background: #2563eb; border: none; border-radius: 8px; font-weight: 600; }
    .btn-success   { background: #059669; border: none; border-radius: 8px; font-weight: 600; }
    .loading-msg   { color: #64748b; font-size: 0.8rem; margin-top: 6px; text-align: center; }
  "))),

  # ── Top bar ────────────────────────────────────────────────────────────────
  div(style = "padding: 1.2rem 1.5rem 0.8rem; border-bottom: 1px solid #2d3148; margin-bottom: 1rem;",
    div(class = "app-title", "[ EMAIL RESPONSE OPTIMIZER ]"),
    div(class = "app-sub",
        "logistic regression · bag-of-words · RAG + OpenRouter · Enron corpus")
  ),

  # ── Main layout ─────────────────────────────────────────────────────────────
  layout_columns(
    col_widths = c(4, 8),
    gap = "1rem",

    # ── Left: Input panel ───────────────────────────────────────────────────
    card(
      card_header("INPUT"),
      card_body(
        textAreaInput("email_text", label = NULL,
                      placeholder = "paste email here...",
                      rows = 11, width = "100%"),

        actionButton("predict_btn", "▶  RUN MODEL",
                     class = "btn-primary w-100 mb-2",
                     style = "letter-spacing: 0.08em;"),

        tags$hr(style = "border-color: #2d3148;"),

        actionButton("revise_btn", "✦  GENERATE AI REVISION",
                     class = "btn-success w-100",
                     style = "letter-spacing: 0.08em;"),

        uiOutput("loading_msg")
      )
    ),

    # ── Right: Results ──────────────────────────────────────────────────────
    layout_columns(
      col_widths = c(5, 7),
      gap = "1rem",

      # Probability score
      card(
        card_header("RESPONSE PROBABILITY"),
        card_body(uiOutput("prob_display"))
      ),

      # Gauge chart
      card(
        card_header("CONFIDENCE GAUGE"),
        card_body(plotlyOutput("gauge", height = "200px"))
      ),

      # Radar chart — spans full width
      card(
        full_screen = TRUE,
        card_header("EMAIL FEATURE RADAR"),
        card_body(plotlyOutput("radar", height = "300px"))
      )
    )
  ),

  # ── AI Revision (full width, appears after clicking Generate) ───────────────
  uiOutput("revision_section")
)

# ── Server ────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # --- Prediction -----------------------------------------------------------
  # reactiveVal lets us explicitly set it to NULL on reset
  result <- reactiveVal(NULL)

  observeEvent(input$predict_btn, {
    req(nchar(trimws(input$email_text)) > 0)
    result(list(
      prob     = predict_prob(input$email_text),
      features = get_features(input$email_text)
    ))
  })

  # Big probability number + badge + tip
  output$prob_display <- renderUI({
    if (is.null(result()))
      return(p("awaiting input...", style = "color:#475569; font-size:0.85rem;"))

    p <- result()$prob
    info <- if (p >= 60) list(cls_num = "prob-high", cls_badge = "badge-high", label = "HIGH",
                               tip = "Clear, direct language. High engagement signal.")
             else if (p >= 35) list(cls_num = "prob-mod", cls_badge = "badge-mod", label = "MODERATE",
                               tip = "Add a direct question or tighten the ask.")
             else               list(cls_num = "prob-low", cls_badge = "badge-low", label = "LOW",
                               tip = "Too vague or too long. Make the ask explicit.")
    tagList(
      div(class = paste("prob-number", info$cls_num), sprintf("%.0f%%", p)),
      div(style = "margin-top: 8px;", span(class = info$cls_badge, info$label)),
      div(class = "tip", info$tip)
    )
  })

  # Gauge chart
  output$gauge <- renderPlotly({
    p <- if (!is.null(result())) result()$prob else 0
    plot_ly(type = "indicator", mode = "gauge+number", value = p,
            number = list(suffix = "%", font = list(size = 24, color = "#e2e8f0")),
            gauge = list(
              axis     = list(range = list(0, 100), tickcolor = "#475569"),
              bar      = list(color = "#38bdf8", thickness = 0.2),
              bgcolor  = "#1e2130", borderwidth = 0,
              steps    = list(list(range = c(0, 35),   color = "#450a0a"),
                              list(range = c(35, 60),  color = "#451a03"),
                              list(range = c(60, 100), color = "#064e3b")),
              threshold = list(line = list(color = "#e2e8f0", width = 2),
                               thickness = 0.8, value = p)
            )) |>
      layout(paper_bgcolor = "rgba(0,0,0,0)", font = list(color = "#e2e8f0"),
             margin = list(t = 20, b = 10, l = 30, r = 30)) |>
      config(displayModeBar = FALSE)
  })

  # Radar / spider chart — shows the 5 key text features normalized to 0–10
  output$radar <- renderPlotly({
    if (is.null(result())) return(plotly_empty())

    f <- result()$features

    # Normalize each feature to a 0–10 scale against reasonable max values
    axes   <- c("Word Count", "Avg Word Length", "Syllable Count",
                 "Stop Words", "Exclamations")
    maxes  <- c(150, 8, 200, 60, 8)   # domain max for each axis
    raw    <- c(as.numeric(f[["Word Count"]]),
                as.numeric(f[["Avg Word Length"]]),
                as.numeric(f[["Syllable Count"]]),
                as.numeric(f[["Stop Word Count"]]),
                as.numeric(f[["Exclamation Marks"]]))
    scores <- pmin(raw / maxes * 10, 10)   # clamp at 10

    # Close the polygon by repeating first point
    axes_closed   <- c(axes,   axes[1])
    scores_closed <- c(scores, scores[1])

    plot_ly(type = "scatterpolar", mode = "lines+markers",
            r = scores_closed, theta = axes_closed,
            fill = "toself",
            fillcolor = "rgba(56,189,248,0.15)",
            line = list(color = "#38bdf8", width = 2),
            marker = list(color = "#38bdf8", size = 6)) |>
      layout(
        polar = list(
          bgcolor = "rgba(0,0,0,0)",
          radialaxis = list(range = c(0, 10), tickfont = list(color = "#64748b"), gridcolor = "#2d3148"),
          angularaxis = list(tickfont = list(color = "#94a3b8", size = 11), gridcolor = "#2d3148")
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        showlegend = FALSE,
        margin = list(t = 30, b = 30, l = 50, r = 50)
      ) |>
      config(displayModeBar = FALSE)
  })

  # --- AI Revision ----------------------------------------------------------
  revision   <- reactiveVal(NULL)
  is_loading <- reactiveVal(FALSE)

  output$loading_msg <- renderUI({
    if (is_loading())
      div(class = "loading-msg", "⏳ retrieving best practices · generating revision...")
  })

  # Clears everything back to the initial state
  observeEvent(input$reset_btn, {
    updateTextAreaInput(session, "email_text", value = "")
    result(NULL)
    revision(NULL)
    is_loading(FALSE)
  })

  # Runs when user clicks GENERATE AI REVISION
  observeEvent(input$revise_btn, {
    req(nchar(trimws(input$email_text)) > 0)
    prob <- if (!is.null(result())) result()$prob else 50
    is_loading(TRUE)
    revision(NULL)
    revision(suggest_revision(input$email_text, prob))
    is_loading(FALSE)
  })

  # Renders the full-width revision card below the main layout
  output$revision_section <- renderUI({
    req(revision())
    rev <- revision()

    change_items <- lapply(rev$changes, function(ch)
      div(class = "change-line",
          span(class = "change-arrow", "→"), ch))

    div(style = "padding: 0 0 1.5rem;",
      layout_columns(
        col_widths = c(7, 5),
        gap = "1rem",

        card(
          card_header("AI-REVISED EMAIL"),
          card_body(div(class = "revised-box", rev$revised_email))
        ),

        card(
          card_header("CHANGES MADE"),
          card_body(
            div(change_items),
            div(class = "rag-note",
                "↳ RAG · all-MiniLM-L6-v2 embeddings · openrouter/owl-alpha"),
            tags$hr(style = "border-color: #2d3148; margin-top: 1rem;"),
            actionButton("reset_btn", "↺  CLEAR & START OVER",
                         class = "w-100",
                         style = "letter-spacing: 0.08em; background: #1e2130;
                                  border: 1px solid #2d3148; color: #94a3b8;
                                  border-radius: 8px; font-weight: 600;")
          )
        )
      )
    )
  })
}

shinyApp(ui, server)
