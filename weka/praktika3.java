package weka;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import static weka.classifiers.lazy.IBk.WEIGHT_NONE;
import static weka.classifiers.lazy.IBk.WEIGHT_SIMILARITY;
import static weka.classifiers.lazy.IBk.WEIGHT_INVERSE;
import static weka.classifiers.lazy.IBk.TAGS_WEIGHTING;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;
import java.util.Random;

public class praktika3 {
	

	    public static void main(String[] args) throws Exception {
	    	
	        ConverterUtils.DataSource source = new ConverterUtils.DataSource("/home/jon/Escritorio/Wekapraktika3/src/weka/balance-scale.arff");  
	        Instances data = source.getDataSet();
	        int kkopurua = data.numInstances();
	       // data.setClassIndex(0);
	        //int kkopurua = Integer.parseInt("10");
	        if(data.classIndex() == -1) {
	        	data.setClassIndex(data.numAttributes()-1);
	        }
	        
	        System.out.println(data.numInstances());
	        System.out.println(data.numAttributes());
	        

	        System.out.println("kNN-algoritmoaren parametro ekorketa");

	        LinearNNSearch[] distantziak = distantziaklortu();
	        
	        System.out.println("Distantziak lortuta");
	        
	        SelectedTag[] tags = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING),new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};

	        System.out.println("Tags lortuta");
	        
	        System.out.println("K kopurua -> " + kkopurua);

	        double maxMeasure = 0.0;
	        int k = 1;
	        int w = 0;
	        int d = 0;
	        IBk knn = new IBk();

	        
	        for(int i=1; i<=kkopurua; i++){
	            knn.setKNN(i);
	            
	            for(int j=0; j<distantziak.length; j++){
	                knn.setNearestNeighbourSearchAlgorithm(distantziak[j]);
	                
	                for(int l=0; l<tags.length; l++) {
	                	try {
	                    knn.setDistanceWeighting(tags[l]);
	                    Evaluation evaluation = new Evaluation(data);
	                    evaluation.crossValidateModel(knn,data,10,new Random(i));                   
	                    if(evaluation.weightedFMeasure() > maxMeasure) {
	                        maxMeasure = evaluation.weightedFMeasure();
	                       // System.out.println(evaluation.weightedFMeasure());
	                        k=i;
	                        w=j;
	                        d=l;
	                    }
	                	} catch(Exception e) {
	                    	System.out.println("Arazoak egon dira: k -> " + i);
	                    }
	                }
	            }
	        }
	        System.out.println("F-MEASURE maximoa -> " + maxMeasure);
	        System.out.println("K hoberena -> " + k);
	        System.out.println("Distantzia mota hoberena -> " + motalortu(w));
	        System.out.println("Tag -> " + d);
	    }
	    



	    private static LinearNNSearch[] distantziaklortu() throws Exception {
	    		    	
	    	 //Euclidean
	    	 LinearNNSearch euclideandistance = new LinearNNSearch();
	         euclideandistance.setDistanceFunction(new EuclideanDistance());;

	         //Manhattan 
	         LinearNNSearch manhattandistance = new LinearNNSearch();
	         manhattandistance.setDistanceFunction(new ManhattanDistance());
	         
	         //Minkowski 
	         LinearNNSearch minkowskidistance = new LinearNNSearch();
	         minkowskidistance.setDistanceFunction(new MinkowskiDistance());

	        return new LinearNNSearch[] {euclideandistance, manhattandistance, minkowskidistance};

	    }
	    
	    private static String motalortu (int w) {
	    	
	    	String mota="";
	    	if (w==0) {
	    		mota="EuclideanDistance";
	    	}
	    	else {
	    		if (w==1) {
	    			mota="ManhattanDistance";
	    		}
	    		else {
	    			mota="MinkowskiDistance";
	    		}
	    	}
	    		
	    	return mota;
	    	
	    }

}
