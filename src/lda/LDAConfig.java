package lda;

import java.io.File;

import config.Config;

public class LDAConfig
{	
	public static int numTrainIters=10;
	public static int numTestIters=10;
	public static boolean SLModel=true; //Short for Save/Load Model
	
	public static String dataPath=Config.dataPath+"lda"+File.separator;
	public static String modelPath=dataPath+"model"+File.separator;
	
	public static String vocabFileName=dataPath+"vocab.txt";
	
	public static String trainCorpusFileName=dataPath+"corpus_train.txt";
	public static String trainLinkFileName=dataPath+"link_train.txt";
	public static String trainClusterFileName=dataPath+"cluster_train.txt";
	
	public static String testCorpusFileName=dataPath+"corpus_test.txt";
	public static String testTrainLinkFileName=dataPath+"link_test_train.txt";
	public static String testTestLinkFileName=dataPath+"link_test_test.txt";
	public static String testTrainClusterFileName=dataPath+"cluster_test_train.txt";
	
	public static String getModelFileName(String model)
	{
		return modelPath+model+"_model.txt";
	}
}
