#include "dirichlet.hpp"

dirichlet::dirichlet(int n){
    alpha=VectorXd::Ones(n);
    meanprecision();
}

dirichlet::dirichlet(VectorXd a){
    setAlpha(a);
}
dirichlet::dirichlet(){
    
}

void dirichlet::setAlpha(VectorXd a){
    alpha=a;
    meanprecision();
}

double dirichlet::log_likelihood(const Ref<const VectorXd>& counts){
    VectorXd log_counts=VectorXd::Zero(counts.size());
    VectorXd log_alpha=VectorXd::Zero(alpha.size());
    VectorXd log_alpha_counts=VectorXd::Zero(alpha.size());
    double csum=counts.sum();
    for(int i=0;i<counts.size();i++){
        log_counts[i]=lgamma(counts[i]+1);
        log_alpha[i]=lgamma(alpha[i]);
        log_alpha_counts[i]=lgamma(alpha[i]+counts[i]);
    }
    double loglike=0.0;
    loglike+=lgamma(csum+1)-log_counts.sum();
    loglike+=lgamma(s) - lgamma(s + csum);
    loglike+=(log_alpha_counts-log_alpha).sum();
    return loglike;
}

void dirichlet::meanprecision(){
    s= alpha.sum();
    m= (1.0f/s)*alpha;
}

void dirichlet::dirichlet_moment_match(const Ref<const MatrixXd>& proportions, const Ref<const MatrixXd>& weigths){
    VectorXd a;
    VectorXd m2;
    VectorXd aok,m2ok;
    double res=0;
    a= average(proportions,weigths,0);
    m2 = average(proportions.cwiseProduct(proportions),weigths,0);
    
    aok = VectorXd(a.size());
    m2ok = VectorXd(m2.size());
    
    int k=0;
    for(int i=0;i<a.size();i++){
        if(a[i]>0){
            aok[k]=a[i];
            m2ok[k]=m2[i];
            k++;
        }else{
            aok.conservativeResize(aok.size()-1);
            m2ok.conservativeResize(m2ok.size()-1);
        }
    }

    res=median((aok - m2ok).cwiseQuotient((m2ok - aok.cwiseProduct(aok))));

    alpha=a*res;
    
}

void dirichlet::dirichlet_moment_match(const Ref<const MatrixXd>& counts){
    MatrixXd norm_sum = counts.rowwise().sum();
    MatrixXd norm_counts(counts.rows(),counts.cols());
    for(int i=0;i<counts.rows();i++){
        if(norm_sum(i,0)!=0){
            norm_counts.row(i)= counts.row(i)*(1/norm_sum(i,0)); // 
        }else{
            continue;
        }
    }
    dirichlet_moment_match(norm_counts,norm_sum);
}

void dirichlet::fit_fixedPoint(MatrixXd& counts,int maxIter,double tol){ //incomplete
    int train=counts.rows();
    int D=counts.cols();
    int iter=0;
    VectorXd old_alp ;
    MatrixXd c;
    VectorXd d;
    double change = 2*tol;
    //counts = counts[sum(counts.A, axis=1) > 0, :]
    MatrixXd auxCounts = counts.rowwise().sum();
    for(int i=0;i<auxCounts.rows();i++){
         if(auxCounts(i,0)<0){
            removeRow(auxCounts,i);
            removeRow(counts,i);
            i--;
         }
    }

    //alpha = array(dirichlet_moment_match(counts)).flatten()
    dirichlet_moment_match(counts);
    c = MatrixXd::Zero(train,D);
    d = VectorXd::Zero(train);

    while(change > tol && iter < maxIter)
    {
        old_alp=alpha;
        for (int i=0;i<train;i++){
            c.row(i)=psi(counts.row(i)+alpha.transpose())-psi(alpha.transpose());
            d[i]= psi(counts.row(i).sum()+alpha.sum())-psi(alpha.sum());
        }
        auxCounts=(c.colwise().sum())*(1.0/d.sum());
        alpha=alpha.cwiseProduct(auxCounts.transpose());
        change = (alpha-old_alp).cwiseAbs().maxCoeff();
        iter++;
    }
    
}

void dirichlet::polya_fit_m(MatrixXd& counts,double tol)
{
    meanprecision();
    VectorXd old_m,a;
    for (int i=0;i<20;i++){
        old_m=m;
        a=s*m;
        for(int j=0;j<counts.cols();j++)
        {
            m(j)= (a(j)*di_pochhammer(a(j),counts.col(j))).sum();
        }
        m/=(m.sum());
        if ((m-old_m).cwiseAbs().maxCoeff() < tol){
            break;
        }
    }
    alpha= s*m;
}


void dirichlet::s_derivatives(MatrixXd& counts, double *g,double *h)
{
    meanprecision();
    *g=-1.0*(di_pochhammer(s,counts.rowwise().sum()).sum());
    *h=-1.0*(tri_pochhammer(s,counts.rowwise().sum()).sum());
    
    for(int k=0;k<counts.cols();k++){
        *g+=m[k]* di_pochhammer(alpha[k],counts.col(k)).sum();
        *h+=std::pow(m[k],2.0)*tri_pochhammer(alpha[k],counts.col(k)).sum();
    }
}
double dirichlet::stable_a2(MatrixXd& counts){
    VectorXd scounts=counts.rowwise().sum();
    double a,ak; 
    m= alpha*(1.0/alpha.sum());
    a = (1.0/6.0)*(scounts*(scounts.array()-1).matrix()*(2*scounts.array()-1).matrix()).sum();
    for(int k=0;k<counts.cols();k++){
        ak = (1.0/6.0)*(counts.col(k)*(counts.col(k).array()-1).matrix()*(2*counts.col(k).array()-1).matrix()).sum();
        if(ak > 0){
            a-=ak/(m[k]*m[k]);
        }
    }
    return a;
}

void dirichlet::polya_fit_s(MatrixXd& counts,double tol)
{
    double h,g,eps,c,a0,a1,a2,b;
    VectorXd old_alpha;
    meanprecision();
    eps= std::numeric_limits<double>::epsilon();
    
    for(int iter=0;iter<10;iter++){
        s_derivatives(counts,&g,&h);
        if (g > eps){
            c = g+s*h;  
            if(c >=0){
                s=INFINITY;
            }else{
                s=s/(1.0+g/(h*s));
            }
        } 
        if(g<-eps){
            
            c = positives(counts)- positives(counts.rowwise().sum());
            
            if(c >0){
                a0 = s*s*h+c;
                a1 = 2.0*s*s*(s*h+g);
                if( abs(2.0*g+h*s) > eps ){
                    a2=s*s*s*(2.0*g+h*s);
                }else{
                    a2= stable_a2(counts);
                }
                b= quad_root(a2,a1,a0);
                s= 1/ ((1 / s) - (g / c) * std::pow((s + b)/b,2));
            }
        }
        old_alpha=alpha;
        alpha=m*s;
        if((old_alpha-alpha).cwiseAbs().maxCoeff()<tol){
            break;
        }

    }

}

void dirichlet::fit_betabinom_minka_alternating(MatrixXd& counts, int maxiter, double tol)
{
    removeNoTrials(counts);
    double change = 2*tol;
    VectorXd alpha_old;
    dirichlet_moment_match(counts);

    for(int iter=0;iter<maxiter && change>tol;iter++){
        alpha_old=alpha;
        polya_fit_m(counts,tol);
        polya_fit_s(counts,tol);
        change = (alpha_old - alpha).cwiseAbs().maxCoeff();
    }
}
