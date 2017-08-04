library(ggplot2)
library(jsonlite)
library(grid)
theme_set(theme_bw(base_size = 14, base_family = "Arial"))

# cv_d<-fromJSON("test_cv_nl/cv_d_stats.json") 

cv_1<-fromJSON("test_cv_nl2/cv_2_stats.json") 
#cv_2<-fromJSON("test_cv_nl2_re/cv_2_stats.json") 
#cv_3<-fromJSON("test_cv_nl2_re_ec/cv_2_stats.json") 

cv<-data.frame(
  ants_1=cv_1$gkappa,
  ants_1_re=cv_2$gkappa,
  ants_1_re_ec=cv_3$gkappa

)

cvv<-stack(cv)

names(cvv)=c('GenKappa','Method')

png('cv_nl.png',width=800,height=400,type='cairo')

ggplot(data=cvv,aes(x=Method,y=GenKappa))+
 geom_boxplot(notch=T)+
 theme_bw()+
 theme(
   axis.text = element_text(face = 'bold', vjust = 0.2, size = 18),
   axis.title = element_text(face = 'bold', vjust = 0.2, size = 20),
   plot.margin = unit(c(0.2,2.8,0.2,0.2), "cm")
   )
