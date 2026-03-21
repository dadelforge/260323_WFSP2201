# Excess mortality analysis – Brussels Region
# Model: log(deaths) ~ constant + annual harmonic
# Envelope: 97.72nd percentile of residuals (≈ +2 SD)
# COVID mask: March 2020 – December 2020 (excluded from fit & residual quantile)

library(ggplot2)

# ── 1. Load & prepare ─────────────────────────────────────────────────────────
df <- read.csv("df_mortality_brussels.csv", sep = ";", header = TRUE,
               col.names = c("week", "deaths"))

df$date <- as.Date(df$week, format = "%d-%m-%y")
df      <- df[order(df$date), ]
df$t    <- seq_len(nrow(df))          # integer week index

# ── 2. COVID mask ─────────────────────────────────────────────────────────────
covid_start <- as.Date("2020-03-01")
covid_end   <- as.Date("2020-12-31")
df$covid    <- df$date >= covid_start & df$date <= covid_end

# ── 3. Log-transform & annual harmonic terms ──────────────────────────────────
T_weeks       <- 365.25 / 7          # ≈ 52.18 weeks per year
df$log_deaths <- log(df$deaths)
df$sin1       <- sin(2 * pi * df$t / T_weeks)
df$cos1       <- cos(2 * pi * df$t / T_weeks)

# ── 4. Fit model on non-COVID weeks only ──────────────────────────────────────
train <- df[!df$covid, ]
model <- lm(log_deaths ~ sin1 + cos1, data = train)
cat("Model summary:\n"); print(summary(model))

# ── 5. Predict for all weeks & compute excess envelope ────────────────────────
df$fitted_log <- predict(model, newdata = df)

# Residuals on training weeks only
train_resid <- df$log_deaths[!df$covid] - df$fitted_log[!df$covid]
thresh      <- quantile(train_resid, 0.9772)
cat(sprintf("\n97.72th percentile of residuals (log scale): %.4f\n", thresh))

df$envelope_log <- df$fitted_log + thresh

# Back-transform to counts
df$fitted   <- exp(df$fitted_log)
df$envelope <- exp(df$envelope_log)

# ── 6. Flag excess weeks ──────────────────────────────────────────────────────
df$excess <- df$deaths > df$envelope

cat(sprintf("Excess weeks (outside COVID mask): %d\n",
            sum(df$excess & !df$covid)))
cat(sprintf("Excess weeks (inside COVID mask):  %d\n",
            sum(df$excess &  df$covid)))

# ── 7. Plot ───────────────────────────────────────────────────────────────────
p <- ggplot(df, aes(x = date)) +

  # COVID exclusion band
  annotate("rect",
           xmin = covid_start, xmax = covid_end + 6,
           ymin = -Inf, ymax = Inf,
           fill = "#BBBBBB", alpha = 0.30) +

  # Excess mortality shading — one rect per week to avoid cross-gap polygons
  geom_rect(data = df[df$excess, ],
            aes(xmin = date, xmax = date + 6, ymin = envelope, ymax = deaths),
            fill = "#D62728", alpha = 0.55, inherit.aes = FALSE) +

  # Observed deaths
  geom_line(aes(y = deaths), colour = "black", linewidth = 0.45) +

  # Seasonal model (back-transformed)
  geom_line(aes(y = fitted), colour = "#1F77B4", linewidth = 1.0) +

  # 97.72 % excess envelope
  geom_line(aes(y = envelope), colour = "#D62728",
            linetype = "dashed", linewidth = 0.8) +

  labs(
    title    = "Weekly all-cause mortality — Brussels Region",
    subtitle = paste0(
      "Blue: seasonal model  |  Red dashed: 97.72 % excess envelope\n",
      "Red shading: weeks above envelope  |  Grey band: COVID-19 mask (Mar\u2013Dec 2020)"
    ),
    x = "Date",
    y = "Deaths per week"
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme_bw(base_size = 12) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey30"))

ggsave("excess_mortality_brussels.png", plot = p,
       width = 12, height = 5, dpi = 150)
cat("\nPlot saved to excess_mortality_brussels.png\n")
