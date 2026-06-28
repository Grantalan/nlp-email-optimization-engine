# ── Email Response Optimizer v2 ───────────────────────────────────────────────

library(shiny)
library(bslib)
library(plotly)
library(reticulate)
library(DT)

tryCatch(
  use_virtualenv("email-optimizer-v2", required = TRUE),
  error = function(e) {
    virtualenv_create("email-optimizer-v2")
    pkgs <- readLines("requirements.txt")
    pkgs <- pkgs[nzchar(trimws(pkgs))]
    virtualenv_install("email-optimizer-v2", packages = pkgs, pip_options = "--quiet")
    use_virtualenv("email-optimizer-v2", required = TRUE)
  }
)
source_python("predict_helper.py")
source_python("rag_helper.py")

# ── UI ────────────────────────────────────────────────────────────────────────

ui <- page_fillable(

  theme = bs_theme(bootswatch = "darkly", base_font = font_google("IBM Plex Mono")),

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
                     white-space: pre-wrap; color: #d1fae5; font-family: 'IBM Plex Mono', monospace;
                     height: 100%; overflow-y: auto; }
    .change-line   { font-size: 0.85rem; color: #94a3b8; padding: 5px 0; border-bottom: 1px solid #2d3148; }
    .change-line:last-child { border-bottom: none; }
    .change-arrow  { color: #38bdf8; font-weight: 700; margin-right: 6px; }
    .rag-note      { font-size: 0.72rem; color: #475569; margin-top: 8px; }
    textarea.form-control { background: #0f1117 !important; color: #e2e8f0 !important;
                            border: 1px solid #2d3148 !important; border-radius: 8px;
                            font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
    .btn-primary   { background: #2563eb; border: none; border-radius: 8px; font-weight: 600; }
    .btn-success   { background: #059669; border: none; border-radius: 8px; font-weight: 600; }
    .loading-msg   { color: #64748b; font-size: 0.8rem; margin-top: 6px; text-align: center; }
    .full-page     { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                     background: #0f1117; z-index: 9999; overflow-y: auto;
                     padding: 1.5rem; box-sizing: border-box; }
    .full-page-header { font-size: 1.4rem; font-weight: 700; letter-spacing: 0.05em;
                        color: #38bdf8; margin-bottom: 0.2rem; }
    .full-page-sub    { font-size: 0.8rem; color: #64748b; margin-bottom: 1.2rem;
                        border-bottom: 1px solid #2d3148; padding-bottom: 0.8rem; }
  "))),

  # Always-visible header
  div(style = "padding: 1.2rem 1.5rem 0.8rem; border-bottom: 1px solid #2d3148; margin-bottom: 1rem;",
    div(class = "app-title", "[ EMAIL RESPONSE OPTIMIZER ]"),
    div(class = "app-sub", "logistic regression · bag-of-words · RAG + OpenRouter · Enron corpus")
  ),

  # Main content — swaps between prediction view and revision view
  uiOutput("page_content")
)

# ── Server ────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  result   <- reactiveVal(NULL)
  revision <- reactiveVal(NULL)
  is_loading <- reactiveVal(FALSE)

  # --- Page router ----------------------------------------------------------
  # Renders either the main prediction screen or the full-screen revision view
  output$page_content <- renderUI({

    if (!is.null(revision())) {

      # ── REVISION SCREEN (full page overlay) ────────────────────────────────
      rev <- revision()

      change_items <- lapply(rev$changes, function(ch)
        div(class = "change-line", span(class = "change-arrow", "→"), ch))

      div(class = "full-page",
        div(class = "full-page-header", "[ AI-SUGGESTED REVISION ]"),
        div(class = "full-page-sub", "RAG · all-MiniLM-L6-v2 · openrouter/owl-alpha"),

        layout_columns(
          col_widths = c(7, 5),
          gap = "1rem",
          style = "height: calc(100vh - 120px);",

          card(
            style = "height: 100%;",
            card_header("REVISED EMAIL"),
            card_body(
              style = "overflow-y: auto; height: calc(100% - 48px);",
              div(class = "revised-box", rev$revised_email)
            )
          ),

          card(
            style = "height: 100%;",
            card_header("CHANGES MADE"),
            card_body(
              style = "display: flex; flex-direction: column; justify-content: space-between; height: calc(100% - 48px);",
              div(div(change_items),
                  div(class = "rag-note", "↳ RAG · all-MiniLM-L6-v2 embeddings · openrouter/owl-alpha")),
              div(
                tags$hr(style = "border-color: #2d3148;"),
                actionButton("reset_btn", "↺  CLEAR & START OVER",
                             class = "w-100",
                             style = "letter-spacing: 0.08em; background: #1e2130;
                                      border: 1px solid #2d3148; color: #94a3b8;
                                      border-radius: 8px; font-weight: 600;")
              )
            )
          )
        )
      )

    } else {

      # ── MAIN PREDICTION SCREEN ─────────────────────────────────────────────
      layout_columns(
        col_widths = c(4, 8),
        gap = "1rem",

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

        layout_columns(
          col_widths = c(5, 7, 5, 7),
          gap = "1rem",

          card(
            card_header("RESPONSE PROBABILITY"),
            card_body(uiOutput("prob_display"))
          ),
          card(
            card_header("CONFIDENCE GAUGE"),
            card_body(plotlyOutput("gauge", height = "200px"))
          ),
          card(
            full_screen = TRUE,
            card_header("WORD COEFFICIENTS  ·  top words from corpus"),
            card_body(plotlyOutput("coef_chart", height = "300px"))
          ),
          card(
            full_screen = TRUE,
            card_header("EMAIL WORD CONTRIBUTIONS  ·  your input words ranked by impact"),
            card_body(DTOutput("contrib_table"), style = "padding: 0.5rem 0 0;")
          )
        )
      )
    }
  })

  # --- Prediction -----------------------------------------------------------
  observeEvent(input$predict_btn, {
    req(nchar(trimws(input$email_text)) > 0)
    result(list(
      prob     = predict_prob(input$email_text),
      features = get_features(input$email_text)
    ))
  })

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

  output$gauge <- renderPlotly({
    p <- if (!is.null(result())) result()$prob else 0
    plot_ly(type = "indicator", mode = "gauge+number", value = p,
            number = list(suffix = "%", font = list(size = 24, color = "#e2e8f0")),
            gauge = list(
              axis  = list(range = list(0, 100), tickcolor = "#475569"),
              bar   = list(color = "#38bdf8", thickness = 0.2),
              bgcolor = "#1e2130", borderwidth = 0,
              steps = list(list(range = c(0, 35),   color = "#450a0a"),
                           list(range = c(35, 60),  color = "#451a03"),
                           list(range = c(60, 100), color = "#064e3b")),
              threshold = list(line = list(color = "#e2e8f0", width = 2),
                               thickness = 0.8, value = p)
            )) |>
      layout(paper_bgcolor = "rgba(0,0,0,0)", font = list(color = "#e2e8f0"),
             margin = list(t = 20, b = 10, l = 30, r = 30)) |>
      config(displayModeBar = FALSE)
  })

  output$coef_chart <- renderPlotly({
    email <- if (!is.null(result())) input$email_text else NULL
    data  <- get_top_coefficients(email, n = 12L)
    words <- unlist(data$word); coefs <- as.numeric(unlist(data$coef))
    in_email <- unlist(data$in_email)
    bar_colors <- ifelse(in_email == "Yes", "#f59e0b", ifelse(coefs > 0, "#34d399", "#f87171"))
    ord <- order(coefs)
    words <- words[ord]; coefs <- coefs[ord]
    bar_colors <- bar_colors[ord]; in_email <- in_email[ord]
    hover_text <- ifelse(in_email == "Yes",
      paste0("<b>", words, "</b>  coef: ", round(coefs, 3), "  ★ in your email"),
      paste0("<b>", words, "</b>  coef: ", round(coefs, 3)))
    plot_ly(x = coefs, y = words, type = "bar", orientation = "h",
            marker = list(color = bar_colors, line = list(width = 0)),
            text = hover_text, hoverinfo = "text") |>
      layout(
        xaxis = list(title = "Model Coefficients", zeroline = TRUE,
                     zerolinecolor = "#475569", zerolinewidth = 1,
                     gridcolor = "#2d3148", tickfont = list(color = "#94a3b8")),
        yaxis = list(title = "", tickfont = list(color = "#e2e8f0", size = 11)),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "IBM Plex Mono", color = "#e2e8f0"),
        margin = list(t = 10, b = 40, l = 10, r = 20)
      ) |>
      config(displayModeBar = FALSE)
  })

  output$contrib_table <- renderDT({
    req(result())
    data <- get_email_word_contributions(input$email_text)
    df   <- data.frame(
      Word        = unlist(data$Word),
      Count       = as.integer(unlist(data$Count)),
      Coefficient = as.numeric(unlist(data$Coefficient)),
      Direction   = unlist(data$Direction),
      stringsAsFactors = FALSE
    )
    datatable(df, rownames = FALSE, height = "100%",
              options = list(pageLength = 20, dom = 't',
                             order = list(list(2, 'desc')),
                             columnDefs = list(list(className = 'dt-center', targets = 1:2)),
                             scrollY = "260px", scrollCollapse = TRUE)) |>
      formatStyle("Direction",
        color = styleEqual(c("Toward Response", "Away From Response", "Not in Model"),
                           c("#34d399", "#f87171", "#64748b"))) |>
      formatStyle("Coefficient",
        background = styleInterval(0, c("#3b0f0f", "#0f3b1f")),
        color      = styleInterval(0, c("#f87171", "#34d399"))) |>
      formatRound(columns = "Coefficient", digits = 4)
  }, server = FALSE)

  # --- AI Revision ----------------------------------------------------------
  output$loading_msg <- renderUI({
    if (is_loading())
      div(class = "loading-msg", "⏳ retrieving best practices · generating revision...")
  })

  observeEvent(input$reset_btn, {
    updateTextAreaInput(session, "email_text", value = "")
    result(NULL)
    revision(NULL)
    is_loading(FALSE)
  })

  observeEvent(input$revise_btn, {
    req(nchar(trimws(input$email_text)) > 0)
    prob <- if (!is.null(result())) result()$prob else 50
    is_loading(TRUE)
    revision(NULL)
    revision(suggest_revision(input$email_text, prob))
    is_loading(FALSE)
  })
}

shinyApp(ui, server)
