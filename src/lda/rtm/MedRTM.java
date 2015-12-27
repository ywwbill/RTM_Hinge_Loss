package lda.rtm;

import java.util.ArrayList;
import java.util.HashMap;

import lda.LDAConfig;
import lda.LDAParam;
import lda.util.LDAResult;
import utility.MathUtil;
import utility.Util;
import cc.mallet.util.MVNormal;

public class MedRTM extends RTM
{
	public ArrayList<HashMap<Integer, Double>> zeta;
	public ArrayList<HashMap<Integer, Double>> lambda;
	
	public void readCorpus(String corpusFileName) throws Exception
	{
		super.readCorpus(corpusFileName);
		for (int doc=0; doc<numDocs; doc++)
		{
			zeta.add(new HashMap<Integer, Double>());
			lambda.add(new HashMap<Integer, Double>());
		}
	}
	
	public void readGraph(String graphFileName, int graphType) throws Exception
	{
		super.readGraph(graphFileName, graphType);
		if (graphType!=TRAIN_GRAPH) return;
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int d : trainEdgeWeights.get(doc).keySet())
			{
				zeta.get(doc).put(d, 0.0);
				lambda.get(doc).put(d, 1.0);
			}
		}
	}
	
	public void sampleNegEdge()
	{
		int numNegEdges=(int)(numTrainEdges*param.negEdgeRatio),u,v;
		for (int i=0; i<numNegEdges; i++)
		{
			u=randoms.nextInt(numDocs);
			v=randoms.nextInt(numDocs);
			while (u==v || corpus.get(u).docLength()==0 || corpus.get(v).docLength()==0 || trainEdgeWeights.get(u).containsKey(v))
			{
				u=randoms.nextInt(numDocs);
				v=randoms.nextInt(numDocs);
			}
			trainEdgeWeights.get(u).put(v, -1);
		}
	}
	
	public void sample(int numIters)
	{
		for (int iteration=1; iteration<=numIters; iteration++)
		{
			if (type==TRAIN)
			{
				optimize();
			}
			
			for (int doc=0; doc<numDocs; doc++)
			{
				weight=new double[trainEdgeWeights.get(doc).size()];
				sampleDoc(doc);
				computeZeta(doc);
				sampleLambda(doc);
			}
			
			computeLogLikelihood();
			perplexity=Math.exp(-logLikelihood/numTestWords);
			
			if (iteration%param.showPLRInterval==0) computePLR();
			Util.println("<"+iteration+">"+"\tLLD: "+logLikelihood+"\tPPX: "+perplexity+"\tPLR: "+PLR);
		}
		
		if (type==TRAIN)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				Util.println(topWords(topic, 10));
			}
		}
	}
	
	//optimize eta by sampling
	public void optimize()
	{
		double sigma[][]=new double[param.numTopics][param.numTopics];
		double vector[]=new double[param.numTopics];
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			sigma[topic][topic]=1.0/(param.nu*param.nu);
		}
		
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int d : trainEdgeWeights.get(doc).keySet())
			{
				for (int topic=0; topic<param.numTopics; topic++)
				{
					vector[topic]=1.0*corpus.get(doc).topicCounts[topic]/corpus.get(doc).docLength()*
							corpus.get(d).topicCounts[topic]/corpus.get(d).docLength();
				}
				for (int t1=0; t1<param.numTopics; t1++)
				{
					for (int t2=0; t2<param.numTopics; t2++)
					{
						sigma[t1][t2]+=param.c*param.c*vector[t1]*vector[t2]/lambda.get(doc).get(d);
					}
				}
			}
		}
		
		sigma=MathUtil.invert(sigma);
		
		double temp[]=new double[param.numTopics];
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int d : trainEdgeWeights.get(doc).keySet())
			{
				for (int topic=0; topic<param.numTopics; topic++)
				{
					temp[topic]+=MathUtil.sqr(param.c)*(lambda.get(doc).get(d)+param.c)/lambda.get(doc).get(d)*
							corpus.get(doc).topicCounts[topic]/corpus.get(doc).docLength()*
							corpus.get(d).topicCounts[topic]/corpus.get(d).docLength();
				}
			}
		}
		
		double mu[]=new double[param.numTopics];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int k=0; k<param.numTopics; k++)
			{
				mu[topic]+=sigma[topic][k]*temp[k];
			}
		}
		
		double sigmaVec[]=new double[param.numTopics*param.numTopics];
		for (int t1=0; t1<param.numTopics; t1++)
		{
			for (int t2=0; t2<param.numTopics; t2++)
			{
				sigmaVec[t1*param.numTopics+t2]=sigma[t1][t2];
			}
		}
		
		eta=MVNormal.nextMVNormal(mu, sigmaVec, randoms);
	}
	
	public void sampleLambda(int doc)
	{
		for (int d : trainEdgeWeights.get(doc).keySet())
		{
			double newValue=MathUtil.sampleIG(1.0/(param.c*Math.abs(zeta.get(doc).get(d))), 1.0);
			lambda.get(doc).put(d, 1.0/newValue);
		}
	}
	
	public double topicUpdating(int doc, int topic, int vocab)
	{
		double score=0.0;
		if (type==TRAIN)
		{
			score=(alpha[topic]+corpus.get(doc).topicCounts[topic])*
					(param.beta+topics.get(topic).vocabCounts[vocab])/
					(param.beta*param.numVocab+topics.get(topic).totalTokens);
		}
		else
		{
			score=(alpha[topic]+corpus.get(doc).topicCounts[topic])*phi[topic][vocab];
		}
		int i=0;
		for (int d : trainEdgeWeights.get(doc).keySet())
		{
			double term1=(param.c*trainEdgeWeights.get(doc).get(d)*(param.c+lambda.get(doc).get(d))*
					eta[topic]*corpus.get(d).topicCounts[topic])/
					(lambda.get(doc).get(d)*corpus.get(doc).docLength()*corpus.get(d).docLength());
			double term2=MathUtil.sqr(param.c)*(MathUtil.sqr(eta[topic]*corpus.get(d).topicCounts[topic])+
					2.0*eta[topic]*corpus.get(d).topicCounts[topic]*weight[i])/
					(2.0*lambda.get(doc).get(d)*MathUtil.sqr(corpus.get(doc).docLength()*corpus.get(d).docLength()));
			score*=Math.exp(term1-term2);
			i++;
		}
		return score;
	}
	
	public void computeZeta(int doc)
	{
		for (int d : trainEdgeWeights.get(doc).keySet())
		{
			double w=computeWeight(doc, d);
			zeta.get(doc).put(d, 1.0-trainEdgeWeights.get(doc).get(d)*w);
		}
	}
	
	public double computeEdgeProb(int doc1, int doc2)
	{
		return Math.exp(-2.0*param.c*Math.max(0.0, 1.0-computeWeight(doc1, doc2)));
	}
	
	public void initVariables()
	{
		super.initVariables();
		zeta=new ArrayList<HashMap<Integer, Double>>();
		lambda=new ArrayList<HashMap<Integer, Double>>();
	}
	
	public MedRTM(LDAParam parameters)
	{
		super(parameters);
	}
	
	public MedRTM(MedRTM rtm, LDAParam parameters)
	{
		super(rtm, parameters);
	}
	
	public MedRTM(String modelFileName, LDAParam parameters) throws Exception
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

		MedRTM RTMTrain=new MedRTM(parameters);
		RTMTrain.readCorpus(LDAConfig.trainCorpusFileName);
		RTMTrain.readGraph(LDAConfig.trainLinkFileName, TRAIN_GRAPH);
		RTMTrain.readGraph(LDAConfig.trainLinkFileName, TEST_GRAPH);
		RTMTrain.sample(LDAConfig.numTrainIters);
		RTMTrain.addResults(trainResults);
		if (LDAConfig.SLModel)
		{
			RTMTrain.writeModel(LDAConfig.getModelFileName(modelName));
		}
		
		MedRTM RTMTest=(LDAConfig.SLModel?
				new MedRTM(LDAConfig.getModelFileName(modelName), parameters):
				new MedRTM(RTMTrain, parameters));
		RTMTest.readCorpus(LDAConfig.testCorpusFileName);
		RTMTest.readGraph(LDAConfig.testTrainLinkFileName, TRAIN_GRAPH);
		RTMTest.readGraph(LDAConfig.testTestLinkFileName, TEST_GRAPH);
		RTMTest.sample(LDAConfig.numTestIters);
		RTMTest.addResults(testResults);
		
		trainResults.printResults(modelName+" Training PLR: ", LDAResult.PLR);
		testResults.printResults(modelName+" Test PLR: ", LDAResult.PLR);
	}
}
