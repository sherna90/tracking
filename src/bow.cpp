#include "bow.h"



Bow::Bow()
{

}

void Bow::readImages(string imagesfile,string clases)
{
    ifstream filename(imagesfile);
    ifstream clases_fs(clases);
    string a,b;
    if(filename.is_open())
    {
        while(getline(filename,a)&&getline(clases_fs,b))
        {

            BowImage data;
            data.setImg(imread(a));
            if(!data.getImg().data)
                cout<<"error imread"<<endl;
            data.setFile(a);
            data.setClase(stoi(b));
            //#### computar####
            trainDescriptors.push_back(data.computeimage());
            //####################
            images.push_back(data);

        }
    }
    else
        return exit(0);
    clases_fs.close();
    filename.close();

}

void Bow::readImage(string imagefile, int clase)
{

    BowImage data=BowImage();
    cout <<imagefile<<endl;
    data.setImg(imread(imagefile));
    if(!data.getImg().data)
        cout<<"error imread"<<endl;
    data.setFile(imagefile);
    data.setClase(clase);
    //#### computar####
    trainDescriptors.push_back(data.computeimage());
    //####################
    images.push_back(data);
}
void Bow::kmeans(int clusters)
{
    if(clusters<1)
        cout<<"error clusters..."<<endl;
    TermCriteria terminate_criterion;
    terminate_criterion.epsilon = FLT_EPSILON;
    BOWKMeansTrainer bowtrainer=BOWKMeansTrainer( clusters, terminate_criterion, 3, KMEANS_PP_CENTERS);
    bowtrainer.add(trainDescriptors);
    vocabulary = bowtrainer.cluster();
}

bool Bow::computeHistograms()
{
    Ptr<SURF> extractor=SURF::create();
    Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce");
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    if(vocabulary.size==0)
        return false;
    else{
        bowDE.setVocabulary( vocabulary );
        for (int i = 0; i < images.size(); ++i) {
            Mat histogram;
            std::vector<KeyPoint> keypoints=images.at(i).getKeypoints();
            bowDE.compute(images.at(i).getImg(), keypoints, histogram);
            //###### normalización ######
            //imgDescriptor /= keypointDescriptors.size().height; código bagofwords.cpp
            histogram*=vocabulary.size().height;
            //######################
            histogram.convertTo(histogram,CV_16S);
            images.at(i).setHistogram(histogram);
            //cout << vocabulary<<endl;
        }
        return true;
    }
}

void Bow::printHistograms()
{
    for (int i = 0; i < images.size(); ++i) {
        cout<<images.at(i).getHistogram() <<endl;
    }
}



