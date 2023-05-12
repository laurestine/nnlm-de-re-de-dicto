library(arm) # transforming data to apply models
library(lmerTest) # mixed effects models
library(lme4) # mixed effect models
library(MuMIn) # evaluating models with conditional, marginal R2
library(patchwork) # combining plots nicely
library(tidyverse) # <3

mycolorscale = c("blue4","#91b653","brown3","slategrey")

# MARK Reading and Preparing Data

quarter1 = read.csv("../csvfiles/q1_processed.csv",header=TRUE,sep=",")
quarter2 = read.csv("../csvfiles/q2_processed.csv",header=TRUE,sep=",")
quarter3 = read.csv("../csvfiles/q3_processed.csv",header=TRUE,sep=",")
quarter4 = read.csv("../csvfiles/q4_processed.csv",header=TRUE,sep=",")

data.raw = tibble(bind_rows(list(quarter1,quarter2,quarter3,quarter4)))

# Turn things into factors if they should be factors:
data.raw %>%
mutate(type = factor(query_type),
    matrix_subject = factor(matrix_NP),
    emb_subject = factor(embedded_noun),
    matrix_verb = factor(matrix_verb),
    emb_verb = factor(embedded_verb),
    det_type = factor(determiner_type),
    matrix_type_fine = factor(matrix_verb_type),
    followup_verb = factor(followup_verb)) ->
data.raw

data.raw %>% 
dplyr::select(type,matrix_subject,emb_subject,matrix_verb,emb_verb,det_type,matrix_type_fine,followup_verb,score) ->
data.raw

# Make column for broad matrix type (ignoring finiteness)
data.raw$matrix_type = ""
data.raw$matrix_type[data.raw$matrix_type_fine %in% c("intensional","intensional_nf")] = "intensional"
data.raw$matrix_type[data.raw$matrix_type_fine %in% c("perceptual")] = "perceptual"
data.raw$matrix_type = factor(data.raw$matrix_type)

# Transform to compute matrix subject bias
data.raw %>%
pivot_wider(names_from="type",values_from="score") ->
data.raw

data.raw %>%
rename(matrix_subj_numscore = matrix_NP, emb_subj_numscore = embedded_NP) ->
data.raw

data.raw$naive_matrix_bias = data.raw$matrix_subj_numscore - data.raw$emb_subj_numscore

# MARK Model Fitting

# Preparing data to fit model
data.raw %>% mutate(
    matrix_subj_rs = arm::rescale(as.numeric(matrix_subject)-1,binary.inputs="-0.5,0.5"),
    matrix_type_rs = arm::rescale(as.numeric(matrix_type)-1,binary.inputs="-0.5,0.5"),
    det_type_rs = arm::rescale(as.numeric(det_type)-1,binary.inputs="-0.5,0.5")) ->
data.raw

contrasts(data.raw$followup_verb) <- contr.helmert(3)

# Our model will have:
# response = naive_matrix_bias
# main effect of det_type, matrix_type and interaction
# main effect of matrix_subject, followup_verb
# random slope of emb_subject by det_type, matrix_type, interaction, matrix subject, followup_verb-correlated
# random slope of matrix_verb by det_type, matrix_subject, followup_verb-correlated
# random slope of emb_verb by det_type, matrix_type, interaction, matrix_subject, followup_verb-correlated

# Random intercepts only (for computing model statistics later)
# (This model only takes about a minute to fit.)
mylmerbase = lmer(naive_matrix_bias ~ det_type_rs * matrix_type_rs + matrix_subj_rs + followup_verb +
    (1 | emb_subject) + (1 | matrix_verb) + (1 | emb_verb), data = data.raw)

# Fitting the full models takes many hours! We have to do it in two rounds.

################################################################################
####### BEGIN COMMENTING HERE TO SKIP MODEL FITTING AND LOAD SAVED MODEL #######
################################################################################

mylmermaxuncor = lmer(naive_matrix_bias ~ det_type_rs * matrix_type_rs + matrix_subj_rs + followup_verb +
   (0 + det_type_rs * matrix_type_rs + matrix_subj_rs|| emb_subject) +
   (1 + followup_verb | emb_subject) +
   (0 + det_type_rs + matrix_subj_rs || matrix_verb) +
   (1 + followup_verb | matrix_verb) +
   (0 + det_type_rs * matrix_type_rs + matrix_subj_rs || emb_verb) +
   (1 + followup_verb | emb_verb),
   data = data.raw,
   control = lmerControl(optimizer="bobyqa"))

saveRDS(mylmermaxuncor, file = "mylmermod.rds")
mylmermaxuncor = readRDS("mylmermod.rds")

fittedVals <- getME(mylmermaxuncor, "theta")

mylmermaxuncor2 = lmer(naive_matrix_bias ~ det_type_rs * matrix_type_rs + matrix_subj_rs + followup_verb +
   (0 + det_type_rs * matrix_type_rs + matrix_subj_rs|| emb_subject) +
   (1 + followup_verb | emb_subject) +
   (0 + det_type_rs + matrix_subj_rs || matrix_verb) +
   (1 + followup_verb | matrix_verb) +
   (0 + det_type_rs * matrix_type_rs + matrix_subj_rs || emb_verb) +
   (1 + followup_verb | emb_verb),
   data = data.raw,
   control = lmerControl(optimizer="bobyqa",
   optCtrl = list(xtol_abs = 1e-8, ftol_abs = 1e-8)),
   start = fittedVals)

saveRDS(mylmermaxuncor2, file = "mylmermod2.rds")

################################################################################
######## END COMMENTING HERE TO SKIP MODEL FITTING AND LOAD SAVED MODEL ########
################################################################################

mylmermaxuncor2 = readRDS("mylmermod2.rds")

# Model statistics
r.squaredGLMM(mylmermaxuncor2,mylmer1)

# MARK Plots for SCiL paper

# Plot in main text
data.raw %>%
group_by(matrix_type,det_type) %>%
summarize(
    medbias = median(naive_matrix_bias),
    roundmedbias = round(medbias,1)) %>%
mutate(
    textx = (0.4*(as.numeric(det_type)-1.5))+as.numeric(matrix_type)) ->
    medlabels

data.raw %>%
ggplot(aes(x=matrix_type,y=naive_matrix_bias,color=det_type)) +
    geom_boxplot() +
    geom_label(data=medlabels,aes(x=textx,y=medbias,label=roundmedbias),color="black",size=3) +
    scale_color_manual(values=mycolorscale[c(1,3)],labels=c('that','a/an')) +
    theme(legend.position="bottom") +
    labs(x="Matrix Verb Type",y="Matrix Subject Bias",color="Determiner") -> matverbanddet_paper

pdf("matverbanddet-scil.pdf",width=3,height=3.8)
print(matverbanddet_paper)
dev.off()

# Plots for appendix
# Matrix subject score
data.raw %>%
    ggplot(aes(x=matrix_subj_numscore)) +
    geom_histogram(binwidth=0.5) +
    coord_cartesian(xlim=c(-30,0),ylim=c(0,100000)) + 
    labs(x="Matrix Subject Score",y="") -> matrixhistogram

# Embedded subject score
data.raw %>%
    ggplot(aes(x=emb_subj_numscore)) +
    geom_histogram(binwidth=0.5)  +
    coord_cartesian(xlim=c(-30,0),ylim=c(0,100000)) + 
    labs(x="Embedded Subject Score",y="") -> embhistogram

# Their shared label
bimodylabel <- ggplot(data.frame(l = "Number of test sentences", x = 1, y = 1)) +
    geom_text(aes(x, y, label = l), angle = 90,size=4) + 
    theme_void() +
    coord_cartesian(clip = "off")

pdf("matrixbimodal-scil.pdf",width=3,height=3.15)
print(bimodylabel + (matrixhistogram/embhistogram) + plot_layout(widths=c(1,11)))
dev.off()

# Effect of main subject
data.raw %>%
group_by(matrix_subject) %>%
summarize(
    medbias = median(naive_matrix_bias),
    roundmedbias = round(medbias,1))  ->
    gendermedlabels

data.raw %>%
ggplot(aes(x=matrix_subject,y=naive_matrix_bias)) +
    geom_boxplot() +
    geom_label(data=gendermedlabels,aes(y=medbias,label=roundmedbias),color="black") +
    labs(x="Matrix Subject",y="Matrix Subject Bias") +
    scale_color_manual(values=mycolorscale[c(1,3)]) -> genderplot

pdf("gender-scil.pdf",width=3,height=3)
print(genderplot)
dev.off()

# Effect of followup verb
data.raw %>%
group_by(followup_verb) %>%
summarize(
    medbias = median(naive_matrix_bias),
    roundmedbias = round(medbias,1))  ->
    followupmedlabels

data.raw %>%
    ggplot(aes(x=followup_verb,y=naive_matrix_bias)) +
    geom_boxplot() +
    geom_label(data=followupmedlabels,aes(y=medbias,label=roundmedbias),color="black") +
    labs(x="Followup Verb",y="Matrix Subject Bias") -> followupplot

pdf("followup-scil.pdf",width=3,height=3)
print(followupplot)
dev.off()

# Effect of syntactic frame and determiner
data.raw %>%
group_by(matrix_type_fine,det_type) %>%
summarize(
    medbias = median(naive_matrix_bias),
    roundmedbias = round(medbias,1)) %>%
mutate(
    textx = (0.4*(as.numeric(det_type)-1.5))+as.numeric(matrix_type_fine)) ->
    framemedlabels

data.raw %>%
ggplot(aes(x=matrix_type_fine,y=naive_matrix_bias,color=det_type)) +
    geom_boxplot() +
    geom_label(data=framemedlabels,aes(x=textx,y=medbias,label=roundmedbias),color="black",size=3) +
    scale_color_manual(values=mycolorscale[c(1,3)],labels=c('that','a/an')) +
    scale_x_discrete(labels=c("verb+'that'", "verb+'to'", "perceptual")) +
    theme(legend.position="bottom") +
    labs(x="Syntactic Frame",y="Matrix Subject Bias",color="Determiner") -> syntacticframe_paper

pdf("syntacticframe-scil.pdf",width=3,height=3.8)
print(syntacticframe_paper)
dev.off()


# Raw effect of indefinite
data.raw %>%
group_by(matrix_subject,emb_verb,followup_verb,emb_subject,matrix_type,det_type) %>%
summarize(mean_bias=mean(naive_matrix_bias)) %>%
pivot_wider(id_cols=c(matrix_subject,emb_verb,followup_verb,emb_subject),names_from=c(matrix_type,det_type),values_from=mean_bias) %>%
mutate(indef_effect = 0.5*(intensional_indefinite + perceptual_indefinite - intensional_deictic - perceptual_deictic)) %>%
ggplot(aes(x=indef_effect)) +
    geom_histogram(position="identity") +
    coord_cartesian(xlim=c(-15,15)) +
    labs(x="Raw effect of 'a/an'",y="") -> indivpairsindef

# Raw effect of intensional
data.raw %>%
group_by(matrix_subject,emb_verb,followup_verb,emb_subject,matrix_type,det_type) %>%
summarize(mean_bias=mean(naive_matrix_bias)) %>%
pivot_wider(id_cols=c(matrix_subject,emb_verb,followup_verb,emb_subject),names_from=c(matrix_type,det_type),values_from=mean_bias) %>%
mutate(intens_effect = 0.5*(intensional_indefinite + intensional_deictic - perceptual_indefinite - perceptual_deictic)) %>%
ggplot(aes(x=intens_effect)) +
    geom_histogram(position="identity") +
    coord_cartesian(xlim=c(-15,15)) +
    labs(x="Raw effect of intensional matrix",y="") -> indivpairsintens

# Raw interaction
data.raw %>%
group_by(matrix_subject,emb_verb,followup_verb,emb_subject,matrix_type,det_type) %>%
summarize(mean_bias=mean(naive_matrix_bias)) %>%
pivot_wider(id_cols=c(matrix_subject,emb_verb,followup_verb,emb_subject),names_from=c(matrix_type,det_type),values_from=mean_bias) %>%
mutate(interaction = (intensional_indefinite - perceptual_indefinite - intensional_deictic + perceptual_deictic)) %>%
ggplot(aes(x=interaction)) +
    geom_histogram(position="identity") +
    coord_cartesian(xlim=c(-15,15)) +
    labs(x="Raw 'a/an'-intensional interaction",y="") -> indivpairsinter

# Their shared label
pairshistylabel <- ggplot(data.frame(l = "Number of test sentence frames", x = 1, y = 1)) +
    geom_text(aes(x, y, label = l), angle = 90,size=4) + 
    theme_void() +
    coord_cartesian(clip = "off")

pdf("indivpairshists-scil.pdf",width=3,height=5.2)
print(pairshistylabel + (indivpairsindef/indivpairsintens/indivpairsinter) + plot_layout(widths=c(1,11)))
dev.off()

# Raw effect of indefinite, separated by followup verb.
data.raw %>%
group_by(matrix_subject,emb_verb,followup_verb,emb_subject,matrix_type,det_type) %>%
summarize(mean_bias=mean(naive_matrix_bias)) %>%
pivot_wider(id_cols=c(matrix_subject,emb_verb,followup_verb,emb_subject),names_from=c(matrix_type,det_type),values_from=mean_bias) %>%
mutate(indef_effect = 0.5*(intensional_indefinite + perceptual_indefinite - intensional_deictic - perceptual_deictic)) %>%
ggplot(aes(x=indef_effect,fill=followup_verb)) +
    geom_histogram(alpha=0.5,position="identity",key_glyph='label') +
    scale_fill_manual(values=mycolorscale) +
    coord_cartesian(xlim=c(-15,15)) +
    theme(legend.position="bottom",legend.spacing.x=unit(0.15,"in")) +
    guides(fill=guide_legend(label.position="bottom"))+
    labs(x="Raw effect of 'a/an'",y="Number of test sentence frames",fill="Followup Verb") -> indivpairsindefcolor

pdf("indivpairsindefcolor-scil.pdf",width=3,height=3.4)
print(indivpairsindefcolor)
dev.off()

# Matrix and embedded subject scores by embedded subject
data.raw %>%
group_by(emb_subject) %>%
summarize(mean_matrix = mean(matrix_subj_numscore),
mean_emb = mean(emb_subj_numscore),
sd_matrix = sd(matrix_subj_numscore),
sd_emb = sd(emb_subj_numscore)) %>%
ggplot(aes(x=reorder(emb_subject,mean_matrix-mean_emb))) +
    geom_errorbar(aes(y=mean_matrix,ymin=mean_matrix-sd_matrix,ymax=mean_matrix+sd_matrix,color="Embedded Subject Score"),alpha=0.7) +
    geom_errorbar(aes(y=mean_emb,ymin=mean_emb-sd_emb,ymax=mean_emb+sd_emb,color="Matrix Subject Score"),alpha=0.7) +
    geom_pointrange(aes(y=mean_matrix,ymin=mean_matrix-sd_matrix,ymax=mean_matrix+sd_matrix,color="Embedded Subject Score"),alpha=0.7) +
    geom_pointrange(aes(y=mean_emb,ymin=mean_emb-sd_emb,ymax=mean_emb+sd_emb,color="Matrix Subject Score"),alpha=0.7) +
    coord_flip() +
    scale_color_manual(values=c(mycolorscale[1:2]),labels=c("Matrix Subject Score","Embedded Subject Score")) +
    theme(legend.position="bottom",legend.title=element_blank()) +
    labs(x="Embedded Subject",y="Score") -> varbyembsubj

pdf("embeddedsubj-scil.pdf",width=6,height=8)
print(varbyembsubj)
dev.off()

# Matrix and embedded subject scores by embedded verb
data.raw %>%
group_by(emb_verb) %>%
summarize(mean_matrix = mean(matrix_subj_numscore),
    mean_emb = mean(emb_subj_numscore),
    sd_matrix = sd(matrix_subj_numscore),
    sd_emb = sd(emb_subj_numscore)) %>%
ggplot(aes(x=reorder(emb_verb,mean_matrix-mean_emb))) +
    geom_errorbar(aes(y=mean_matrix,ymin=mean_matrix-sd_matrix,ymax=mean_matrix+sd_matrix,color="Embedded Subject Score"),alpha=0.7) +
    geom_errorbar(aes(y=mean_emb,ymin=mean_emb-sd_emb,ymax=mean_emb+sd_emb,color="Matrix Subject Score"),alpha=0.7) +
    geom_pointrange(aes(y=mean_matrix,ymin=mean_matrix-sd_matrix,ymax=mean_matrix+sd_matrix,color="Embedded Subject Score"),alpha=0.7) +
    geom_pointrange(aes(y=mean_emb,ymin=mean_emb-sd_emb,ymax=mean_emb+sd_emb,color="Matrix Subject Score"),alpha=0.7) +
    coord_flip() +
    scale_color_manual(values=c(mycolorscale[1:2]),labels=c("Matrix Subject Score","Embedded Subject Score")) +
    theme(legend.position="bottom",legend.title=element_blank()) +
    labs(x="Embedded Verb",y="Score") -> varbyembverbs

pdf("embeddedverbs-scil.pdf",width=6,height=8)
print(varbyembverbs)
dev.off()

# Random variation by matrix verb
data.raw %>%
group_by(matrix_verb,matrix_type) %>%
summarize(mean_matrix = mean(matrix_subj_numscore),
    mean_emb = mean(emb_subj_numscore),
    sd_matrix = sd(matrix_subj_numscore),
    sd_emb = sd(emb_subj_numscore)) %>%
mutate(y_color = fct_recode(matrix_type, "intensional" = "black","perceptual"="brown3")) %>%
arrange(mean_matrix-mean_emb) -> matverbsummary

matverbsummary %>%
ggplot(aes(x=reorder(matrix_verb,mean_matrix-mean_emb))) +
    geom_errorbar(aes(y=mean_matrix,ymin=mean_matrix-sd_matrix,ymax=mean_matrix+sd_matrix,color="Embedded Subject Score"),alpha=0.7) +
    geom_errorbar(aes(y=mean_emb,ymin=mean_emb-sd_emb,ymax=mean_emb+sd_emb,color="Matrix Subject Score"),alpha=0.7) +
    geom_pointrange(aes(y=mean_matrix,ymin=mean_matrix-sd_matrix,ymax=mean_matrix+sd_matrix,color="Embedded Subject Score"),alpha=0.7) +
    geom_pointrange(aes(y=mean_emb,ymin=mean_emb-sd_emb,ymax=mean_emb+sd_emb,color="Matrix Subject Score"),alpha=0.7) +
    coord_flip() +
    scale_color_manual(values=c(mycolorscale[1:2]),labels=c("Matrix Subject Score","Embedded Subject Score")) +
    theme(legend.position="bottom",axis.text.y = element_text(colour=matverbsummary$y_color),legend.title=element_blank()) +
    labs(x="Matrix Verb",y="Score") -> varbymatrixverbs

pdf("matrixverbs-scil.pdf",width=6,height=8)
print(varbymatrixverbs)
dev.off()
