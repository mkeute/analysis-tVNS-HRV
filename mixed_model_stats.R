library(readxl)
library(ggplot2)
library(lmerTest)
library(dplyr)
library(gridExtra)
library(xtable)
library(ggthemes)
library(rstudioapi)
pth=getActiveDocumentContext()$path
setwd(dirname(pth))
df=read_xlsx("hrv_results.xlsx")
df$corRSA = atanh(df$corRSA) #convert corRSA to Fisher-z
df$PNN50 = with(df,log(PNN50/(100-PNN50))) #convert PNN50 to log-odds
df$time[df$time == "pre"] = "a_pre"
df$time[df$time == "stim"] = "b_stim"
df$time[df$time == "post"] = "c_post"

df$SD1SD2 = 1/df$SD1SD2

df$tVNS = as.factor(ifelse(df$condition == "sham", "sham", "tVNS"))
df$condition[df$condition == "sham"] = "a_sham"
df$condition[df$condition == "einatmen"] = "b_einatmen"
df$condition[df$condition == "ausatmen"] = "c_ausatmen"

tbscaled = c("HR","SDNN","RMSSD","PNN50","HF", "LFHF","SD1SD2")
statst = data.frame()
statsp = data.frame()
posthoct = data.frame()
posthocp = data.frame()
for(t in tbscaled){
  eval(parse(text = paste("df$y = df$", t, sep = "")))
  if(t == "HR"){
    df$yrel = NA
    df %>% group_by(subjects, location, condition) %>% mutate(yrel = (y-y[time == "a_pre"]),  logyrel = log10(y)-log10(y[time ==  "a_pre"])) -> df
    
    svg("HR.svg")
    print(
      ggplot(df, aes(x = time, y = y, color = tVNS)) + geom_boxplot(width = .2,position = position_dodge(width = .3),outlier.size = 0)+
        geom_point(alpha = .2,position = position_jitterdodge(dodge.width = .3, jitter.width = .2))+
        ylab("HR[bpm]")+
        scale_x_discrete(breaks = c("a_pre", "b_stim", "c_post"),limits = c("a_pre", "b_stim", "c_post"), label = c("pre","stim", "post"))  +
        theme_clean(base_size = 20)
    )
    try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
    
    svg("HRrel.svg")
    print(
      ggplot(df, aes(x = time, y = yrel, color = tVNS)) + geom_boxplot(width = .2,position = position_dodge(width = .3),outlier.size = 0)+
        geom_point(alpha = .2,position = position_jitterdodge(dodge.width = .3, jitter.width = .2))+
        ylab("change to baseline [bpm]")+
        scale_x_discrete(breaks = c("b_stim", "c_post"),limits = c("b_stim", "c_post"), label = c("b_stim", "c_post"))  + 
        theme_clean(base_size = 20)
    )
    try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
    
    svg("HRgrid.svg")
    print(
      ggplot(df, aes(x = time, y = yrel, color = condition)) + geom_boxplot(width = .2,position = position_dodge(width = .3),outlier.size = 0)+
        geom_point(alpha = .2,position = position_jitterdodge(dodge.width = .3, jitter.width = .08))+
        ylab("change to baseline [bpm]")+
        scale_color_discrete(limits = c("a_sham", "b_einatmen", "c_ausatmen"),breaks = c("a_sham", "b_einatmen", "c_ausatmen"), label = c("sham", "inhalation-locked", "exhalation-locked"))+
        scale_x_discrete(breaks = c("b_stim", "c_post"),limits = c("b_stim", "c_post"), label = c("stim", "post"))  + theme_clean(base_size = 20) + facet_grid(rows = vars(location), cols = vars(ear))
    )
    try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
    
  }
  
  if(!is.element(t, c("HR", "corRSA", "PNN50", "logRSA"))){
    df$y = 10*log10(df$y)
  }
  df$yrel = NA
  df %>% group_by(subjects, location, condition) %>% mutate(yrel = (y-y[time == "a_pre"])) -> df
  df[paste(t,"rel",sep="")] = 100*df$yrel

  pdf(paste(t, ".pdf", sep=""))
  print(
  ggplot(df, aes(x = time, y = yrel, color = tVNS)) + geom_boxplot(width = .2,position = position_dodge(width = .3),outlier.size = 0)+
    geom_point(alpha = .2,position = position_jitterdodge(dodge.width = .3, jitter.width = .2))+
    ylab("change to baseline")+
      # stat_summary(geom = "boxplot", position = position_dodge(width = .4)) + 
      scale_x_discrete(breaks = c("b_stim", "c_post"),limits = c("b_stim", "c_post"), label = c("b_stim", "c_post"))  + theme_clean(base_size = 20)
  )
  try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
  

  
  ix = !is.na(df$yrel) & !is.infinite(abs(df$yrel)) & df$time != "pre"
  m1 = anova(lmer("yrel ~ time * tVNS  + ear +location+(1|subjects)+ (1+condition|subjects)+ (1+time|subjects) +  (1+location|subjects)", data = df[ix,]))
 
  
  statst["time",t]  = m1["time","F value"]
  statsp["time",t] = m1["time","Pr(>F)"]
  
  
  statst["tVNS",t]  =  m1["tVNS","F value"]
  statsp["tVNS",t] = m1["tVNS","Pr(>F)"]
  
  statst["location",t]  =  m1["location","F value"]
  statsp["location",t] = m1["location","Pr(>F)"]
  
  statst["ear",t]  =  m1["ear","F value"]
  statsp["ear",t] = m1["ear","Pr(>F)"]

  statst["time x tVNS",t]  =  m1["time:tVNS","F value"]
  statsp["time x tVNS",t] = m1["time:tVNS","Pr(>F)"]
  
  m1 = summary(lmer("y ~  tVNS + (1|subjects)+  (1+condition|subjects)+(1+location|subjects)", data = df[df$time == "a_pre" & !is.infinite(df$y),]))
  posthoct["tVNS vs. sham during pre",t]  =  m1$coefficients["tVNStVNS","t value"]
  posthocp["tVNS vs. sham during pre",t] = m1$coefficients["tVNStVNS","Pr(>|t|)"]
  posthoct["tVNS & sham vs. baseline during pre",t]  =  m1$coefficients["(Intercept)","t value"]
  posthocp["tVNS & sham vs. baseline during pre",t] = m1$coefficients["(Intercept)","Pr(>|t|)"]
  
  m1 = summary(lmer("yrel ~  tVNS + (1|subjects)+  (1+condition|subjects)+(1+location|subjects)", data = df[df$time == "b_stim" & !is.infinite(df$yrel),]))
  posthoct["tVNS vs. sham during stim",t]  =  m1$coefficients["tVNStVNS","t value"]
  posthocp["tVNS vs. sham during stim",t] = m1$coefficients["tVNStVNS","Pr(>|t|)"]
  posthoct["tVNS & sham vs. baseline during stim",t]  =  m1$coefficients["(Intercept)","t value"]
  posthocp["tVNS & sham vs. baseline during stim",t] = m1$coefficients["(Intercept)","Pr(>|t|)"]
 
  m1 = summary(lmer("yrel ~  tVNS + (1|subjects)+  (1+condition|subjects) + (1+location|subjects)", data = df[df$time == "c_post" & !is.infinite(df$yrel),]))
  posthoct["tVNS vs. sham during post",t]  =  m1$coefficients["tVNStVNS","t value"]
  posthocp["tVNS vs. sham during post",t] = m1$coefficients["tVNStVNS","Pr(>|t|)"]
  posthoct["tVNS & sham vs. baseline during post",t]  =  m1$coefficients["(Intercept)","t value"]
  posthocp["tVNS & sham vs. baseline during post",t] = m1$coefficients["(Intercept)","Pr(>|t|)"]
  
  m1 = anova(lmer("yrel ~  ear*location*condition + (1|subjects)+  (1+location|subjects)", data = df[df$time == "b_stim" & df$tVNS == "tVNS"& !is.infinite(df$yrel),]))
  
  posthoct["ear during tVNS",t]  =  m1["ear","F value"]
  posthocp["ear during tVNS",t] = m1["ear","Pr(>F)"]
  
  posthoct["resp.locking during tVNS",t]  =  m1["ear","F value"]
  posthocp["resp.locking during tVNS",t] = m1["ear","Pr(>F)"]
  
  posthoct["location during tVNS",t]  =  m1["location","F value"]
  posthocp["location during tVNS",t] = m1["location","Pr(>F)"]
  
  posthoct["ear x location during tVNS",t]  =  m1["ear:location","F value"]
  posthocp["ear x location during tVNS",t] = m1["ear:location","Pr(>F)"]
  
  posthoct["resp.locking x location during tVNS",t]  =  m1["location:condition","F value"]
  posthocp["resp.locking x location during tVNS",t] = m1["location:condition","Pr(>F)"]
  
  posthoct["resp.locking x ear during tVNS",t]  =  m1["ear:condition","F value"]
  posthocp["resp.locking x ear during tVNS",t] = m1["ear:condition","Pr(>F)"]
  
  posthoct["resp.locking x ear x location during tVNS",t]  =  m1["ear:location:condition","F value"]
  posthocp["resp.locking x ear x location during tVNS",t] = m1["ear:location:condition","Pr(>F)"]
  
  
  
  m1 = anova(lmer("yrel ~  ear*location*condition + (1|subjects)+  (1+condition|subjects)+(1+location|subjects) ", data = df[df$time == "c_post" & df$tVNS == "tVNS"& !is.infinite(df$yrel),]))
  
  posthoct["ear after tVNS",t]  =  m1["ear","F value"]
  posthocp["ear after tVNS",t] = m1["ear","Pr(>F)"]
  
  posthoct["resp.locking after tVNS",t]  =  m1["ear","F value"]
  posthocp["resp.locking after tVNS",t] = m1["ear","Pr(>F)"]
  
  posthoct["location after tVNS",t]  =  m1["location","F value"]
  posthocp["location after tVNS",t] = m1["location","Pr(>F)"]
  
  posthoct["ear x location after tVNS",t]  =  m1["ear:location","F value"]
  posthocp["ear x location after tVNS",t] = m1["ear:location","Pr(>F)"]
  
  posthoct["resp.locking x location after tVNS",t]  =  m1["location:condition","F value"]
  posthocp["resp.locking x location after tVNS",t] = m1["location:condition","Pr(>F)"]
  
  posthoct["resp.locking x ear after tVNS",t]  =  m1["ear:condition","F value"]
  posthocp["resp.locking x ear after tVNS",t] = m1["ear:condition","Pr(>F)"]

  posthoct["resp.locking x ear x location after tVNS",t]  =  m1["ear:location:condition","F value"]
  posthocp["resp.locking x ear x location after tVNS",t] = m1["ear:location:condition","Pr(>F)"]
  }

xtable(statst, type = "latex", file = "ttable.tex")

xtable(statsp, type = "latex", file = "filename2.tex", digits = 3)

xtable(posthoct, type = "latex", file = "ttable.tex")

xtable(posthocp, type = "latex", file = "filename2.tex", digits = 3)

for_cor = as.matrix(df[,tbscaled])
for_cor = for_cor[!is.infinite(for_cor[,"PNN50"]),]
cormat = cor(for_cor)
library(Hmisc)
cortst = rcorr(for_cor)
cormat[!is.na(cortst) & cortst$P >= .05] = NA
pdf("correlations.pdf")
fields::image.plot(cormat, legend.only=F, axes = F, lwd = .5)
title("correlations between scores")
mtext(text=rownames(cormat), side=1, line=0.3, at=seq(from = 0, to =1, length.out = dim(cormat)[1]), las=2, cex=0.8)
mtext(text=colnames(cormat), side=2, line=0.3, at=seq(from = 0, to =1, length.out = dim(cormat)[2]), las=1, cex=0.8)
try(dev.off(), silent = T);try(dev.off(), silent = T);try(dev.off(), silent = T)
