library(ggplot2)
library(jsonlite)
library(grid)
theme_set(theme_bw(base_size = 14, base_family = "Arial"))


# cv_d<-fromJSON("test_cv_nl/cv_d_stats.json") 
# cv_1<-fromJSON("test_cv_nl/cv_1_stats.json") 
#cv_2<-fromJSON("test_cv_nl/cv_2_stats.json")
#cv_3<-fromJSON("test_cv_ants_ln3/cv_2_stats.json")

cv_beta_none<-fromJSON('test_cv_nl/cv_beta_none_stats.json')
cv_beta_05  <-fromJSON('test_cv_nl/cv_beta_0.5_stats.json')
cv_beta_10  <-fromJSON('test_cv_nl/cv_beta_1.0_stats.json')
cv_beta_new <-fromJSON('test_cv_nl/cv_beta_new_stats.json')
cv_nuyl_new <-fromJSON('test_cv_nl2/cv_beta_none_stats.json')


cv<-data.frame(
    beta_none=cv_beta_none$gkappa,
    beta_05=  cv_beta_05$gkappa,
    beta_10=  cv_beta_10$gkappa,
    beta_new= cv_beta_new$gkappa,
    beta_none2=cv_nuyl_new$gkappa
    )

cvv<-stack(cv)

names(cvv)=c('GenKappa','Method')

png('cv_beta.png',width=800,height=400,type='cairo')

ggplot(data=cvv,aes(x=Method,y=GenKappa))+
 geom_boxplot(notch=T)+
 theme_bw()+
 theme(
   axis.text = element_text(face = 'bold', vjust = 0.2, size = 18),
   axis.title = element_text(face = 'bold', vjust = 0.2, size = 20),
   plot.margin = unit(c(0.2,2.8,0.2,0.2), "cm")
   )
