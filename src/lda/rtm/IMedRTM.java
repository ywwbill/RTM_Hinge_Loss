package lda.rtm;

import java.io.BufferedReader;
import java.io.FileReader;

import lda.util.LDADoc;
import lda.util.LDAResult;
import utility.MathUtil;
import lda.LDAConfig;
import lda.LDAParam;

public class IMedRTM extends MedRTM
{
	public double _alpha;
	
	public int clusterSize;
	public int clusterNo[];
	public int clusterTopicCounts[][];
	public int clusterCounts[];
	
	public void readClusters(String clusterFileName) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(clusterFileName));
		String line,seg[];
		clusterSize=0;
		clusterNo=new int[numDocs];
		while ((line=br.readLine())!=null)
		{
			clusterSize++;
			seg=line.split(" ");
			for (int i=0; i<seg.length; i++)
			{
				if (seg[i].length()>0)
				{
					clusterNo[Integer.valueOf(seg[i])]=clusterSize-1;
				}
			}
		}
		br.close();
		
		clusterTopicCounts=new int[clusterSize][param.numTopics];
		clusterCounts=new int[clusterSize];
		for (int i=0; i<numDocs; i++)
		{
			LDADoc doc=corpus.get(i);
			int no=clusterNo[i];
			for (int topic=0; topic<param.numTopics; topic++)
			{
				clusterTopicCounts[no][topic]+=doc.topicCounts[topic];
				clusterCounts[no]+=doc.topicCounts[topic];
			}
		}
	}
	
	public void sampleDoc(int docIdx)
	{
		int word,oldTopic,newTopic;
		double topicScores[]=new double[param.numTopics];
		LDADoc doc=corpus.get(docIdx);
		
		int i=0;
		for (int d : trainEdgeWeights.get(docIdx).keySet())
		{
			weight[i]=computeWeight(docIdx, d);
			i++;
		}
		
		int no=-1;
		if (clusterSize>0)
		{
			no=clusterNo[docIdx];
		}
		
		int interval=getSampleInterval();
		for (int token=0; token<doc.docLength(); token+=interval)
		{
			word=doc.getWord(token);
			oldTopic=doc.getTopicAssign(token);
			if (topics.get(oldTopic).totalTokens==0) continue;
			
			doc.topicCounts[oldTopic]--;
			topics.get(oldTopic).totalTokens--;
			topics.get(oldTopic).vocabCounts[word]--;
			if (no!=-1)
			{
				clusterTopicCounts[no][oldTopic]--;
				clusterCounts[no]--;
			}
			i=0;
			for (int d : trainEdgeWeights.get(docIdx).keySet())
			{
				weight[i]-=eta[oldTopic]/doc.docLength()*
						corpus.get(d).topicCounts[oldTopic]/corpus.get(d).docLength();
				i++;
			}
			
			for (int topic=0; topic<param.numTopics; topic++)
			{
				topicScores[topic]=topicUpdating(docIdx, topic, word);
			}
			
			newTopic=MathUtil.selectDiscrete(topicScores);
			
			doc.setTopicAssign(token, newTopic);
			doc.topicCounts[newTopic]++;
			topics.get(newTopic).totalTokens++;
			topics.get(newTopic).vocabCounts[word]++;
			if (no!=-1)
			{
				clusterTopicCounts[no][newTopic]++;
				clusterCounts[no]++;
			}
			i=0;
			for (int d : trainEdgeWeights.get(docIdx).keySet())
			{
				weight[i]+=eta[newTopic]/doc.docLength()*
						corpus.get(d).topicCounts[newTopic]/corpus.get(d).docLength();
				i++;
			}
		}
	}
	
	public double topicUpdating(int doc, int topic, int vocab)
	{
		int no=-1;
		if (clusterSize>0)
		{
			no=clusterNo[doc];
		}
		double ratio=1.0/param.numTopics;
		if (no!=-1)
		{
			ratio=(clusterTopicCounts[no][topic]+_alpha)/(clusterCounts[no]+_alpha*param.numTopics);
		}
		double score=0.0;
		if (type==TRAIN)
		{
			score=(param.alphaSum*ratio+corpus.get(doc).topicCounts[topic])*
					(param.beta+topics.get(topic).vocabCounts[vocab])/
					(param.beta*param.numVocab+topics.get(topic).totalTokens);
		}
		else
		{
			score=(param.alphaSum*ratio+corpus.get(doc).topicCounts[topic])*phi[topic][vocab];
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
	
	public void computeTheta()
	{
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				int no=-1;
				if (clusterSize>0)
				{
					no=clusterNo[doc];
				}
				double ratio=1.0/param.numTopics;
				if (no!=-1)
				{
					ratio=(clusterTopicCounts[no][topic]+_alpha)/(clusterCounts[no]+_alpha*param.numTopics);
				}
				theta[doc][topic]=(param.alphaSum*ratio+corpus.get(doc).topicCounts[topic])/
						(param.alphaSum+getSampleSize(corpus.get(doc).docLength()));
			}
		}
	}
	
	public void initVariables()
	{
		super.initVariables();
		_alpha=param._alphaSum/param.numTopics;
	}
	
	public IMedRTM(LDAParam parameters)
	{
		super(parameters);
	}
	
	public IMedRTM(IMedRTM rtm, LDAParam parameters)
	{
		super(rtm, parameters);
	}
	
	public IMedRTM(String modelFileName, LDAParam parameters) throws Exception
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

		IMedRTM RTMTrain=new IMedRTM(parameters);
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
		
		IMedRTM RTMTest=(LDAConfig.SLModel?
				new IMedRTM(LDAConfig.getModelFileName(modelName), parameters):
				new IMedRTM(RTMTrain, parameters));
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
