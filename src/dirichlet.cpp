#include "../include/dirichlet.hpp"

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

double dirichlet::log_likelihood(VectorXd counts){
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

void dirichlet::dirichlet_moment_match(MatrixXd proportions, MatrixXd weigths){
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

void dirichlet::dirichlet_moment_match(MatrixXd counts){
    MatrixXd norm_sum = counts.rowwise().sum();
    
    for(int i=0;i<counts.rows();i++){
        if(norm_sum(i,0)!=0){
            counts.row(i)= counts.row(i)*(1/norm_sum(i,0)); //  
        }else{
            continue;
        }
    }
    dirichlet_moment_match(counts,norm_sum);
}

void dirichlet::fit_fixedPoint(MatrixXd counts,int maxIter,double tol){ //incomplete
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