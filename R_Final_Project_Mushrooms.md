Mushroom Edibility
================
Brooke Seibert
3/19/2019

### Executive Summary

This report will explore the process of analyzing different mushroom characteristics to develop a system to classify different mushrooms as being edible or poisonous with the aim to reduce the chance of mushroom hunters misclassifying and consuming poisonous mushrooms. There is no simple rule for determining the edibility of a mushroom. Despite the risk, mushroom hunting is growing in popularity. Therefore, data was collected from the Audubon Society Field Guide to North American Mushrooms in attempt to facilitate identifying foraged mushrooms as being edible or poisonous. Available records cover 23 different gilled mushroom species from the Agaricus and the Lepiota Mushroom Family. This data on mushroom characteristics for thousands of mushroom observations went through a variety of different analytical techniques. Multiple models were developed to determine if there is a way mushrooms can be classified as edible or poisonous based on other characteristics. Models included a Cross-Validation Classification Tree, a Pruned Classification Tree, and two different Multinomial Logistic Regressions were developed. All models performed well with extremely high accuracy rates in classifying the observations, so selecting the best model comes down to implementation. The regression models are too complex to allow easy implementation. For the classification trees, the Cross-Validation Classification Tree used 2 more variables than the Pruned Classification Tree. This means the Cross-Validation Classification Tree has a greater chance for being able to classify a wider range of mushrooms. Therefore, the Cross-Validation Classification Tree was selected as being the best model with this dataset to best classify mushrooms as being poisonous or edible. In addition, data was analyzed to uncover if there are certain mushroom characteristics that only indicate a poisonous or edible mushroom. Specific mushroom characteristics that only edible mushroom observations had include odors of almonds or anise, red or orange gill colors, buff or purple spore print colors, rooted stalks, and stalk colors below the ring of gray, red, or orange. Characteristics that only applied to the poisonous mushroom observations include buff or green gill colors, green or yellow spore print colors, and stalk colors below the ring of buff, cinnamon, or yellow. Odors pertaining only to poisonous mushrooms include creosote, could, musty, pungent, spicy, and fishy.

**Business Questions**

-   Which analytical model can best classify poisonous and edible mushrooms?
-   Are certain mushroom characteristics only indicative of a mushroom being poisonous?
-   Are certain mushroom characteristics only indicative of a mushroom being edible?

### Data Understanding & Preparation

**Data Source**

The dataset was found through Kaggle, formatted as a CSV file with 8,124 records and 23 fields. The dataset originated from the UCI Machine Learning repository's mushroom dataset developed in 1987. (<https://www.kaggle.com/uciml/mushroom-classification/version/1#_=_>). The mushroom records correspond to 23 gilled mushrooms species in the Agaricus and Lepiota Mushroom Family from the Audubon Society Field Guide to North American Mushrooms.

**Variable Dictionary** ![optional caption text](VariableDictionary.png) **Data Preparation**

Before analysis can be started, the data went through a data preparation stage. Data cleaning began by creating friendly names for the variables. Running some basic statistics revealed that all variables are categorical. It was also revealed that the Veil Type, Veil Color, and Gill Attachment variables only had a single categorical level represented in the dataset. This inhibits the usefulness of these variables in classification models. Therefore, all three of these variables were excluded from the dataset, as well as any records with missing values. Finally the data was partitioned allowing 50% of the data for training any models and 50% of the data for testing any models.

### Anlaysis

**Exploratory Analysis**

In total the dataset contains 8,124 observations for 23 different variables. All variables are categorical. Even the target variable is a binary variable classifying each mushroom record as being poisonous or edible. Overall, 51.80% of the observations belong to edible mushrooms and 48.20% of the observations belong to poisonous. The Variable Cross Tabulation Matrix below clearly displays the category distributions for each variable. The distribution of the target variable among the independent variables is also visible by color. Class “e” in red is for edible mushrooms and class “p” in blue is for poisonous mushrooms.

*Variable Cross-Tabulation Matrix* ![](CrossTabluationMatrix.png)

**Cross-Validation Classification Tree**

The cross-validation classification tree model was built utilizing all independent variables to classify the target mushroom classifications. The model was first developed with the training data and then performance was measured by applying and comparing model performance with the testing data. The resulting classification tree includes 3 nodes. The model’s important variables in order of highest importance first include: Odor, Gill Color, Stalk Surface Above Ring, Stalk Surface Below Ring, Gill Size, and Bruises. Visuals of this model are included below. The resulting accuracy rate of this classification tree is 99.48%. The classification rate for a mushroom to be correctly classified as edible is 99.01% and 100% for correctly classifying a poisonous mushroom. The cross-validation size plot and the cross validation tree plot were then developed to help determine beneficial ways to prune the classification tree. These plots are used to help determine the number of nodes and trees to utilize when pruning the tree to compare how the models fit the data.

*Cross-Validation Classification Tree Flexdashboard Display* ![](cvtree%20slide.png)

**Pruned Classification Tree**

The pruned classification tree has 5 terminal nodes. Variables in use include Odor, Spore Print Color, Stalk Color Below Ring, and Stalk Root. A confusion matrix was then designed after the classification tree was ran with the data set aside for testing. The resulting accuracy rate for the tree is 99.88%. Both the Pruned Classification Tree and the Cross-Validation Classification Tree performed very well in correctly classifying the dataset’s poisonous and edible mushrooms. By pruning the classification tree the accuracy rate increased from 99.48% to 99.88% and the amount of important variables for the model reduced to 4. The other tree had 6 important variables.

*Pruned Classifiction Tree Flexdashboard Display* ![](pruned%20tree%20slide.png)

**Multinominal Logistic Regression**

Originally a Multinomial Logistic Regression to classify poisonous and edible mushrooms employing all independent variables. This produced a model that when evaluated with the testing data, had an accuracy rate of 99.93%. However, the model is overly complex with the large quantity of categorical independent variables. Therefore, another model was built. The Revised Multinomial Logistic Regression only included the independent variables that either of the two Classification Tree Models found to be important variables. When the testing data was applied to this model, the Revised Multinomial Logistic Regression had an accuracy rate of 1. There were no false negatives or false positives in the model’s confusion matrix.

*Multinominal Logistic Regression Flexdashboard Display* ![](regression%20slide.png)

**Analysis of Mushroom Characteristics** Variables deemed important through the classification modeling were further analyzed. To test for associations between pairs of variables, Chi Square testing was applied. Based on results, all instances reveal there to be association between the pairs of variables. With variable associations confirmed, the dataset was split into a dataframe containing only the poisonous mushroom observations and a dataframe containing only the edible mushroom observations. Major findings are listed below, followed by screenshots of the flexdashboard as its scrolls down through the different utilized data visualizations.

*Odor*
• Only edible: almonds or anise
• Only poisonous: creosote, foul, musty, pungent, spicy, fishy
• Odorless mushrooms were more commonly edible

*Gill Color*
• Only edible: red or orange
• Only poisonous: buff (most common) or green
• Edible counts all &lt; 1000, poisonous counts all &lt; 2000

*Stalk Surface Above & Below Ring*
• Similar distirubuton for both variables
• No unique characteristics for being edible or poisonous
• Mostly edible: smooth above or below ring (~3,500 instances)
• But smooth stalk surface was commonly poisonous too
• Lowest representation in general were scaly stalk surfaces

*Gill Size*
• Binary where broad or narrow gills could be edible or not
• Edible: vastly more commonly broad (~4000)
• Poisonous: more commonly narrow

*Bruises*
• Binary where bruising is not unique to a target
• Edible mushrooms more commonly bruised
• Most poisonous mushrooms could not bruise (&gt;3000)

*Spore Print Color*
• Lower cateogry counts, below 2000 per level
• Only edible: buff or purple, though with low counts
• Only poisonous: green or yellow, also with low counts
• Most commonly edible: black or brown
• Most commonly poisonous: chocolate or white

*Stalk Color Below Ring*
• White stalks were most common in general
• Only edible: gray (most common), red, orange
• Only poisonous: buff (most common), cinnamon, yellow

*Stalk Root*
• All category counts were low, below 2000
• Only edible: rooted stalks
• Bulbous stalk roots were most common in general
• Missing stalk roots were more commonly poisonous

*Flexdashboard Display for Mushroom Characteristics* ![](edible.png) ![](poisonous.png)

### Results

**Which analytical model can best classify poisonous and edible mushrooms?**

*The Cross-Validation Classification Tree*

All classification models performed well, with high accuracy. However, both Multinomial Logistic Regressions were extremely complex with structure. The classification trees will be easier to implement on new data. As the accuracy rates for both models is high, choosing the best model comes down to implementation. Of the classification trees, the Cross-Validation Classification Tree used 2 more variables than the Pruned Classification Tree. This means the Cross-Validation Classification Tree has a greater chance for being able to classify a wider range of mushrooms. Therefore, the Cross-Validation Classification Tree was selected as being the best model with this dataset to best classify mushrooms as being poisonous or edible.

**Are certain mushroom characteristics only indicative of a mushroom being poisonous?**

• *Odors:* creosote, foul, musty, pungent, spicy, fishy

• *Gill Colors:* buff, green

• *Spore Print Colors:* green, yellow

• *Stalk Color Below Ring:* buff, cinnamon, yellow

**Are certain mushroom characteristics only indicative of a mushroom being edible?**

• *Odors:* almonds, anise

• *Gill Colors:* red, orange

• *Spore Print Color:* buff, purple

• *Stalk Color Below Ring:* gray, red, orange

• *Stalk Root:* rooted

### Conclusion

Data on mushroom characteristics for thousands of mushroom observations went through a variety of different analytical techniques. Multiple models were developed to determine if there is a way mushrooms can be classified as edible or poisonous based on other characteristics. Models included a Cross-Validation Classification Tree, a Pruned Classification Tree, and two different Multinomial Logistic Regressions were developed. All models performed well with extremely high accuracy rates in classifying the observations, so selecting the best model comes down to implementation. The regression models are too complex to allow easy implementation. For the classification trees, the Cross-Validation Classification Tree used 2 more variables than the Pruned Classification Tree. This means the Cross-Validation Classification Tree has a greater chance for being able to classify a wider range of mushrooms. Therefore, the Cross-Validation Classification Tree was selected as being the best model with this dataset to best classify mushrooms as being poisonous or edible. In addition, data was analyzed to uncover if there are certain mushroom characteristics that only indicate a poisonous or edible mushroom. Specific mushroom characteristics that only edible mushroom observations had include odors of almonds or anise, red or orange gill colors, buff or purple spore print colors, rooted stalks, and stalk colors below the ring of gray, red, or orange. Characteristics that only applied to the poisonous mushroom observations include buff or green gill colors, green or yellow spore print colors, and stalk colors below the ring of buff, cinnamon, or yellow. Odors pertaining only to poisonous mushrooms include creosote, could, musty, pungent, spicy, and fishy.

### Appendix: R Code

**Complete Flexdashboard R Code**

``` r
#yeti
library(tidyverse)
library(broom)
library(glmnet)
library(caret)
library(ISLR)
library(janitor)
library(stringr)
library(rsample)
library(ggplot2)
library(GGally)
library(gridExtra)
library(dplyr)
library(MASS)
library(plotROC)
library(randomForest)
library(rpart)
library(rpart.plot)
library(partykit)
library(nnet)
library(tree)
library(reshape)
library(flexdashboard)
theme_set(theme_bw())
df <- readr::read_csv("mushrooms.csv")
#develop friendly variable names
colnames(df)[colnames(df)=="class"] <- "Class"
colnames(df)[colnames(df)=="cap-shape"] <- "CapShape"
colnames(df)[colnames(df)=="cap-surface"] <- "CapSurface"
colnames(df)[colnames(df)=="cap-color"] <- "CapColor"
colnames(df)[colnames(df)=="bruises"] <- "Bruises"
colnames(df)[colnames(df)=="odor"] <- "Odor"
colnames(df)[colnames(df)=="gill-attachment"] <- "GillAttachment"
colnames(df)[colnames(df)=="gill-spacing"] <- "GillSpacing"
colnames(df)[colnames(df)=="gill-size"] <- "GillSize"
colnames(df)[colnames(df)=="gill-color"] <- "GillColor"
colnames(df)[colnames(df)=="stalk-shape"] <- "StalkShape"
colnames(df)[colnames(df)=="stalk-root"] <- "StalkRoot"
colnames(df)[colnames(df)=="stalk-surface-above-ring"] <- "StalkSurfaceAboveRing"
colnames(df)[colnames(df)=="stalk-surface-below-ring"] <- "StalkSurfaceBelowRing"
colnames(df)[colnames(df)=="stalk-color-above-ring"] <- "StalkColorAboveRing"
colnames(df)[colnames(df)=="stalk-color-below-ring"] <- "StalkColorBelowRing"
colnames(df)[colnames(df)=="veil-type"] <- "VeilType"
colnames(df)[colnames(df)=="veil-color"] <- "VeilColor"
colnames(df)[colnames(df)=="ring-number"] <- "RingNumber"
colnames(df)[colnames(df)=="ring-type"] <- "RingType"
colnames(df)[colnames(df)=="spore-print-color"] <- "SporePrintColor"
colnames(df)[colnames(df)=="population"] <- "Population"
colnames(df)[colnames(df)=="habitat"] <- "Habitat"
# Change all variables from characters into factors
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], as.factor)
old_df <- df
# Remove VeilType (only 1 level: p = partical)
#nlevels(df$VeilType)
df <- df[-c(17)]
# Remove VeilColor (vastly 1 level split fairly evenly: w = white)
df <- df[-c(17)]
# Remove GillAttachment (vastly 1 level split fairly evently: f = free)
df <- df[-c(7)]
# Remove any rows with NULLs
df <- na.omit(df)
set.seed(298329)
inTraining <- createDataPartition(df$Class, p = .5, list = F)
training <- df[inTraining, ]
testing  <- df[-inTraining, ]
ep <- "51.80%"
valueBox(ep, color = "lightseagreen")
pp <- "48.20%"
valueBox(pp, color = "indianred")
pp <- "8124"
valueBox(pp, color = "lightgray")
total_variables <- length(old_df)
valueBox(total_variables, color = "lightgray")
cv_tree <- rpart::rpart(Class ~ ., data = training)
plot(as.party(cv_tree))
predict_tree <- predict(cv_tree, testing, positive = "e", type = "class")
#summary(cv_tree)
#variable importance: Odor-28, GillColor-16, StalkSurfaceAboveRing-15, StalkSurfaceBelowRing-15, GillSize-13, Bruises-13
#node 1: 4062, node 2: 2159, node 3: 1903
#Primary splits: Odor, GillColor, StalkSurfaceAboveRing, StalkSurfaceBelowRing, GillSize
#Surrogate splits: GillColor, StalkSurfaceAboveRing, StalkSurfaceBelowRing, GillSize, Bruises
#confusionMatrix(as.factor(predict_tree),as.factor(testing$Class))
# confusion matrix visual
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y      <- c(0, 2104, 1937, 21)
c <- data.frame(TClass, PClass, Y)
ggplot(data =  c, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "lightseagreen", high = "indianred") +
  theme_bw() + theme(legend.position = "none")+ ggtitle("Cross Validation Tree Confusion Matrix") + xlab("True Class") + ylab("Predicted Class")
gauge(99.48, min = 0, max = 100, symbol = '%', gaugeSectors(
  success = c(80, 100), warning = c(40, 79), danger = c(0,39), color = "lightseagreen"
))
#plots
set.seed(3)
a_tree <- tree(Class ~ ., data = training)
cv_tree <- cv.tree(a_tree ,FUN=prune.misclass )
#names(cv_tree )
plot(cv_tree$size, cv_tree$dev, type="b")
plot(cv_tree$k, cv_tree$dev, type="b")
prune_tree <- prune.misclass(a_tree,best=5)
plot(prune_tree)
text(prune_tree, pretty =0)
predict_prune <- predict(prune_tree, testing, positive = "e",type="class")
#summary(prune_tree)
#confusionMatrix(as.factor(predict_prune),as.factor(testing$Class))
# confusion matrix visual
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y      <- c(0, 2104, 1953, 5)
b <- data.frame(TClass, PClass, Y)
ggplot(data =  b, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "lightseagreen", high = "indianred") +
  theme_bw() + theme(legend.position = "none")+ ggtitle("Pruned Classification Tree Confusion Matrix") + xlab("True Class") + ylab("Predicted Class")
gauge(99.88, min = 0, max = 100, symbol = '%', gaugeSectors(
  success = c(80, 100), warning = c(40, 79), danger = c(0,39), color = "lightseagreen"
))
# Basic Multinominal Logistic Regression with Training Data
train_mlr <- multinom(Class ~ ., data = training)
summary(train_mlr)
s_mlr_train <- summary(train_mlr)$coefficients/summary(train_mlr)$standard.errors
s_mlr_train
# 2-tailed z test
z_mlr_train <- (1 - pnorm(abs(s_mlr_train), 0, 1)) * 2
z_mlr_train
# extract the coefficients from the model and exponentiate
oec <- exp(coef(train_mlr))
oec
# predicted probabilities
pp <- fitted(train_mlr)
# predict on testing data
predicted_scores <- predict (train_mlr, testing, "probs")
predict_mlr <- predict(train_mlr, testing, positive = "e")
#confusionMatrix(as.factor(predict_mlr),as.factor(testing$Class))
# confusion matrix visual
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y      <- c(0, 2104, 1955, 3)
d <- data.frame(TClass, PClass, Y)
ggplot(data =  d, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "lightseagreen", high = "indianred") +
  theme_bw() + theme(legend.position = "none")+ ggtitle("Original Multinominal Logistic Regression Confusion Matrix") + xlab("True Class") + ylab("Predicted Class")
gauge(99.93, min = 0, max = 100, symbol = '%', gaugeSectors(
  success = c(80, 100), warning = c(40, 79), danger = c(0,39), color = "lightseagreen"
))
# Basic Multinominal Logistic Regression with Training Data
train_mlr_2 <- multinom(Class ~ Odor+GillColor+StalkSurfaceAboveRing+StalkSurfaceBelowRing+GillSize+Bruises+SporePrintColor+StalkColorBelowRing+StalkRoot, data = training)
summary(train_mlr_2)
s_mlr_train_2 <- summary(train_mlr_2)$coefficients/summary(train_mlr_2)$standard.errors
s_mlr_train_2
# 2-tailed z test
z_mlr_train_2 <- (1 - pnorm(abs(s_mlr_train), 0, 1)) * 2
z_mlr_train_2
# extract the coefficients from the model and exponentiate
opc <- exp(coef(train_mlr_2))
opc
# predicted probabilities
pp_2 <- fitted(train_mlr_2)
# predict on testing data
predicted_scores_2 <- predict (train_mlr_2, testing, "probs")
predict_mlr_2 <- predict(train_mlr_2, testing, positive = "e")
#confusionMatrix(as.factor(predict_mlr_2),as.factor(testing$Class))
# confusion matrix visual
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y      <- c(0, 2104, 1958, 0)
g <- data.frame(TClass, PClass, Y)
ggplot(data =  g, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "lightseagreen", high = "indianred") +
  theme_bw() + theme(legend.position = "none")+ ggtitle("Revised Multinominal Logistic Regression Confusion Matrix") + xlab("True Class") + ylab("Predicted Class")
gauge(100, min = 0, max = 100, symbol = '%', gaugeSectors(
  success = c(80, 100), warning = c(40, 79), danger = c(0,39), color = "lightseagreen"
))
# edible dataframe
e_df <- df[df$Class == "e",]
# poisonous dataframe
p_df <- df[df$Class == "p",]
#ODOR
e_odor <- ggplot(e_df, aes(Odor,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Odor")+ ylim(0, 4000)
e_odor
#GILL COLOR
e_gillcolor <- ggplot(e_df, aes(GillColor,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Gill Color")+ ylim(0, 4000)
e_gillcolor
#STALK SURFACE ABOVE RING
e_ssar <- ggplot(e_df, aes(StalkSurfaceAboveRing,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Stalk Surface Above Ring")+ ylim(0, 4000)
e_ssar
#STALK SURFACE BELOW RING
e_ssbr <- ggplot(e_df, aes(StalkSurfaceBelowRing,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Stalk Surface Below Ring")+ ylim(0, 4000)
e_ssbr
#GILL SIZE
e_gillsize <- ggplot(e_df, aes(GillSize,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Gill Size")+ ylim(0, 4000)
e_gillsize
#BRUISES
e_bruises <- ggplot(e_df, aes(Bruises,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Bruises")+ ylim(0, 4000)
e_bruises
#SPORE PRINT COLOR
e_spcolor <- ggplot(e_df, aes(SporePrintColor,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Spore Print Color")+ ylim(0, 4000)
e_spcolor
#STALK COLOR BELOW RING
e_scbr <- ggplot(e_df, aes(StalkColorBelowRing,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Stalk Color Below Ring")+ ylim(0, 4000)
e_scbr
#STALK ROOT
e_sroot <- ggplot(e_df, aes(StalkRoot,..count.. )) + geom_bar(fill = "lightseagreen")+ theme(text = element_text(size=9)) + ggtitle("Edible Mushrooms by Stalk Root")+ ylim(0, 4000)
e_sroot
#ODOR
p_odor <- ggplot(p_df, aes(Odor,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Odor")+ ylim(0, 4000)
p_odor
#GILL COLOR
p_gillcolor <- ggplot(p_df, aes(GillColor,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Gill Color")+ ylim(0, 4000)
p_gillcolor
#STALK SURFACE ABOVE RING
p_ssar <- ggplot(p_df, aes(StalkSurfaceAboveRing,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Stalk Surface Above Ring")+ ylim(0, 4000)
p_ssar
#STALK SURFACE BELOW RING
p_ssbr <- ggplot(p_df, aes(StalkSurfaceBelowRing,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Stalk Surface Below Ring")+ ylim(0, 4000)
p_ssbr
#GILL SIZE
p_gillsize <- ggplot(p_df, aes(GillSize,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Gill Size")+ ylim(0, 4000)
p_gillsize
#BRUISES
p_bruises <- ggplot(p_df, aes(Bruises,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Bruises")+ ylim(0, 4000)
p_bruises
#SPORE PRINT COLOR
p_spcolor <- ggplot(p_df, aes(SporePrintColor,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Spore Print Color")+ ylim(0, 4000)
p_spcolor
#STALK COLOR BELOW RING
p_scbr <- ggplot(p_df, aes(StalkColorBelowRing,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Stalk Color Below Ring")+ ylim(0, 4000)
p_scbr
#STALK ROOT
p_sroot <- ggplot(p_df, aes(StalkRoot,..count.. )) + geom_bar(fill = "indianred")+ theme(text = element_text(size=9)) + ggtitle("Poisonous Mushrooms by Stalk Root")+ ylim(0, 4000)
p_sroot
chisq.test(df$Class, df$Odor) 
chisq.test(df$Class, df$GillColor) 
chisq.test(df$Class, df$StalkSurfaceAboveRing) 
chisq.test(df$Class, df$StalkSurfaceBelowRing) 
chisq.test(df$Class, df$GillSize)
chisq.test(df$Class, df$Bruises)
chisq.test(df$Class, df$SporePrintColor) 
chisq.test(df$Class, df$StalkColorBelowRing) 
chisq.test(df$Class, df$StalkRoot) 
knitr::include_graphics("1.png")
knitr::include_graphics("2.png")
```
