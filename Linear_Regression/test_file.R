#Define Working Directory
setwd("C:/Users/User/OneDrive/DS/R studio/Projects/Linear Regression")
getwd()

#Load Packages
install.packages("tidyverse")
install.packages("glmnet")
library(tidyverse)
library(glmnet)

data <- read.csv("insurance.csv")

#Prüfen ob NA Werte vorhanden sind -> Keine
sum(is.na(data))

#Die Zielvariable wird ans Ende geschoben

#Die Lasso Regression wird ausgeführt um zu ermitteln, welche Präditkor Variablen verwendet werden sollten, um die Linerare Regression durchzuführen
#Alternativen für die Prädiktor Bewertung wären XXX
#In der Lasso Regression wird eine Kreuzvalidierung durchgeführt um den optimalen Regulierungsparameter zu bestimmen. Dieser dient der Lasso Regression die optimalen Prädiktorvariablen zu bestimmen.

# Beispiel-Datensatz

# Zielvariable
zielvariable <- "charges"

# 2. Daten aufteilen (80% Training, 20% Test)
set.seed(123)  # Für Reproduzierbarkeit
train_index <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#Es sollte ein Vorwärtsintegration durchgeführt werden, jedoch nicht erfolgreich

# Startmodell: leeres Modell
start_model <- lm(as.formula(paste("charges", "~ 1")), data = train_data)

# Volles Modell: alle Variablen
full_model <- lm(as.formula(paste("charges", "~ .")), data = train_data)

# Vorwärtsselektion durchführen
forward_selection <- step(start_model, 
                          direction = "forward", 
                          scope = list(lower = start_model, upper = full_model))

# Zusammenfassung des ausgewählten Modells
summary(forward_selection)

# 7. Vorhersage auf den Testdaten
predictions <- predict(forward_selection, newdata = test_data)

# 8. Modellbewertung
actuals <- test_data$charges
mse <- mean((predictions - actuals)^2)  # Mittlerer quadratischer Fehler
cat("Mean Squared Error on Test Data:", mse, "\n")

#Die Prädiktorvariablen werden jetzt ermittelt in den einfach Variablen mit starker Signifiakz berücksichtigt werden

#Smoker Variable in Binäre Werte umwandeln
data$smoker <- ifelse(data$smoker == "yes", 1, 0)

# 80% der Daten für das Training, 20% für das Testen
#set.seed(123) # Für Reproduzierbarkeit
n_train <- round(0.8 * nrow(data))
train_indices <- sample(1:nrow(data), n_train)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]


# Vorläufige lineare Regression mit allen Variablen auf den Trainingsdaten
full_model <- lm(charges ~ ., data = train_data)

# Zusammenfassung des Modells, um p-Werte zu erhalten
summary(full_model)


#Saving R-squared
r_sq_0 <- summary(full_model)$r.squared
#predict data on test set
prediction_0 <- predict(full_model, newdata = test_data)
#calculating the residuals
residuals_0 <- test_data$charges - prediction_0
#calculating Root Mean Squared Error
rmse_0 <- sqrt(mean(residuals_0^2))




# Neue Lineare Regression nur mit den zuvor identifizieren Signifikanten Variablen
new_model <- lm(charges ~ age + bmi + children + smoker, data = train_data)

# Zusammenfassung des Modells, um p-Werte zu erhalten
summary(new_model)

#Saving R-squared
r_sq_1 <- summary(new_model)$r.squared
#predict data on test set
prediction_1 <- predict(new_model, newdata = test_data)
#calculating the residuals
residuals_1 <- test_data$charges - prediction_1
#calculating Root Mean Squared Error
rmse_1 <- sqrt(mean(residuals_1^2))

#Modelbewertung
print(paste("R² für das erste Modell mit allen Variablen: ", round(r_sq_0, 4)))
print(paste("RMSE Wert, also der Kontrollwert, für das erste Modell mit allen Variablen: ", rmse_0))
print(paste("R² für das zweite Modell mit allen Variablen: ", round(r_sq_1, 4)))
print(paste("RMSE Wert, also der Kontrollwert, für das zweite Modell mit allen Variablen: ", rmse_1))

# influential observations
# Da der Datensatz recht klein ist wird versucht mithilfe des Cook-Abstandes zu ermitteln ob es Datenpunkte gibt die einen besonders großen Einfluss auf die Funktion haben.
cooks_D <- cooks.distance(full_model)
hat_values <- hatvalues(full_model)
# Da keine Werte im Model den allgemein gebräuchlichen Schwellenwert von Cooks D überschreiten kann man sasgen, dass es kein Datenpunkte gibt die einen überaus großen Einfluss auf die Gesamtfunktion haben
std_resid <- rstandard(full_model)
sum(cooks_D > 0.08)
