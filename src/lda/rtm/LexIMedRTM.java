package lda.rtm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;

import cc.mallet.optimize.LimitedMemoryBFGS;
import lda.LDAConfig;
import lda.LDAParam;
import lda.rtm.util.LexIMedRTMFunction;
import lda.util.LDAResult;
import utility.MathUtil;
import utility.Util;

public class LexIMedRTM extends IMedRTM
{
	public ArrayList<HashMap<Integer, Integer>> wordCounts;
	
	public double tau[];
	
	public void readCorpus(String corpusFileName) throws Exception
	{
		super.readCorpus(corpusFileName);
		for (int doc=0; doc<numDocs; doc++)
		{
			HashMap<Integer, Integer> tempCount=new HashMap<Integer, Integer>();
			for (int i=0; i<corpus.get(doc).docLength(); i++)
			{
				int token=corpus.get(doc).getWord(i);
				if (!tempCount.containsKey(token))
				{
					tempCount.put(token, 1);
				}
				else
				{
					int freq=tempCount.get(token);
					tempCount.put(token, freq+1);
				}
			}
			wordCounts.add(tempCount);
		}
	}
	
	public void optimize()
	{
		LexIMedRTMFunction optimizable=new LexIMedRTMFunction(this);
		LimitedMemoryBFGS lbfgs=new LimitedMemoryBFGS(optimizable);
		try
		{
			lbfgs.optimize();
		}
		catch (Exception e)
		{
			return;
		}
		for (int topic=0; topic<param.numTopics; topic++)
		{
			eta[topic]=optimizable.parameters[topic];
		}
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			tau[vocab]=optimizable.parameters[vocab+param.numTopics];
		}
	}
	
	public double computeWeight(int doc1, int doc2)
	{
		double weight=0.0;
		for (int topic=0; topic<param.numTopics; topic++)
		{
			weight+=eta[topic]*corpus.get(doc1).topicCounts[topic]/corpus.get(doc1).docLength()*
					corpus.get(doc2).topicCounts[topic]/corpus.get(doc2).docLength();
		}
		for (int token : wordCounts.get(doc1).keySet())
		{
			if (wordCounts.get(doc2).containsKey(token))
			{
				weight+=tau[token]*wordCounts.get(doc1).get(token)/corpus.get(doc1).docLength()*
						wordCounts.get(doc2).get(token)/corpus.get(doc2).docLength();
			}
		}
		return weight;
	}
	
	public void readModel(String modelFilename) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(modelFilename));
		Util.readMatrix(br, phi);
		Util.readVector(br, alpha);
		Util.readVector(br, eta);
		Util.readVector(br, tau);
		br.close();
	}
	
	public void writeModel(String modelFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(modelFileName));
		Util.writeMatrix(bw, phi);
		Util.writeVector(bw, alpha);
		Util.writeVector(bw, eta);
		Util.writeVector(bw, tau);
		bw.close();
	}
	
	public void initVariables()
	{
		super.initVariables();
		wordCounts=new ArrayList<HashMap<Integer, Integer>>();
		tau=new double[param.numVocab];
	}
	
	public LexIMedRTM(LDAParam parameters)
	{
		super(parameters);
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			tau[vocab]=randoms.nextGaussian(0.0, MathUtil.sqr(param.nu));
		}
	}
	
	public LexIMedRTM(LexIMedRTM rtm, LDAParam parameters)
	{
		super(rtm, parameters);
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			tau[vocab]=randoms.nextGaussian(0.0, MathUtil.sqr(param.nu));
		}
	}
	
	public LexIMedRTM(String modelFileName, LDAParam parameters) throws Exception
	{
		super(modelFileName, parameters);
	}
	
	public static void main(String args[]) throws Exception
	{
		String seg[]=Thread.currentThread().getStackTrace()[1].getClassName().split("\\.");
		String modelName=seg[seg.length-1];
		LDAParam parameters=new LDAParam(LDAConfig.vocabFileName);
		LDAResult trainResults=new LDAResult();
		LDAResult testResults=new LDAResult();

		LexIMedRTM RTMTrain=new LexIMedRTM(parameters);
		RTMTrain.readCorpus(LDAConfig.trainCorpusFileName);
		RTMTrain.readGraph(LDAConfig.trainLinkFileName, TRAIN_GRAPH);
		RTMTrain.readClusters(LDAConfig.trainClusterFileName);
		RTMTrain.readGraph(LDAConfig.trainLinkFileName, TEST_GRAPH);
		RTMTrain.sample(LDAConfig.numTrainIters);
		RTMTrain.addResults(trainResults);
		if (LDAConfig.SLModel)
		{
			RTMTrain.writeModel(LDAConfig.getModelFileName(modelName));
		}
		
		LexIMedRTM RTMTest=(LDAConfig.SLModel?
				new LexIMedRTM(LDAConfig.getModelFileName(modelName), parameters):
				new LexIMedRTM(RTMTrain, parameters));
		RTMTest.readCorpus(LDAConfig.testCorpusFileName);
		RTMTest.readGraph(LDAConfig.testTrainLinkFileName, TRAIN_GRAPH);
		RTMTest.readClusters(LDAConfig.testTrainClusterFileName);
		RTMTest.readGraph(LDAConfig.testTestLinkFileName, TEST_GRAPH);
		RTMTest.sample(LDAConfig.numTestIters);
		RTMTest.addResults(testResults);
		
		trainResults.printResults(modelName+" Training PLR: ", LDAResult.PLR);
		testResults.printResults(modelName+" Test PLR: ", LDAResult.PLR);
	}
}
