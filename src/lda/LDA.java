package lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;

import cc.mallet.util.Randoms;
import lda.util.LDADoc;
import lda.util.LDAResult;
import lda.util.LDATopic;
import lda.util.LDAWord;
import utility.Util;
import utility.MathUtil;

public class LDA
{	
	public static final int TRAIN=0;
	public static final int TEST=1;
	
	public LDAParam param;
	
	public static Randoms randoms;
	
	public double alpha[];

	public int numDocs;
	public int numWords;
	public int numTestWords;
	public int type;
	
	public ArrayList<LDADoc> corpus;
	public ArrayList<LDATopic> topics;
	
	public double theta[][];
	public double phi[][];
	
	public double logLikelihood;
	public double perplexity;
	
	public void readCorpus(String corpusFileName) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(corpusFileName));
		String line;
		while ((line=br.readLine())!=null)
		{
			corpus.add(new LDADoc(line, param.numTopics, param.numVocab));
		}
		br.close();
		
		numDocs=corpus.size();
		numWords=0;
		for (int doc=0; doc<numDocs; doc++)
		{
			numWords+=corpus.get(doc).docLength();
		}
		
		theta=new double[numDocs][param.numTopics];
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics.add(new LDATopic(param.numVocab));
		}
		
		//In training, we use all words
		//In test, we use half of words to infer topic distribution and other half to evaluate
		for (LDADoc doc : corpus)
		{
			int interval=getSampleInterval();
			for (int token=0; token<doc.docLength(); token+=interval)
			{
				int topic=randoms.nextInt(param.numTopics);
				doc.setTopicAssign(token, topic);
				doc.topicCounts[topic]++;
				topics.get(topic).totalTokens++;
				
				int word=doc.getWord(token);
				topics.get(topic).vocabCounts[word]++;
			}
		}
		
		getNumTestWords();
		
		Util.println(this.getClass().getName()+"\t#Docs: "+numDocs+"\t#Topics:"+param.numTopics+
				"\t#Vocab: "+param.numVocab+"\talphaSum="+param.alphaSum);
	}
	
	public void sample(int numIters)
	{
		for (int iteration=1; iteration<=numIters; iteration++)
		{
			for (int doc=0; doc<numDocs; doc++)
			{
				sampleDoc(doc);
			}
			computeLogLikelihood();
			perplexity=Math.exp(-logLikelihood/numTestWords);
			Util.println("<"+iteration+">"+"\tLog-LLD: "+logLikelihood+"\tPPX: "+perplexity);
		}
		
		if (type==TRAIN)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				Util.println(topWords(topic, 10));
			}
		}
	}
	
	public void sampleDoc(int docIdx)
	{
		int word,oldTopic,newTopic;
		double topicScores[]=new double[param.numTopics];
		LDADoc doc=corpus.get(docIdx);
		
		int interval=getSampleInterval();
		for (int token=0; token<doc.docLength(); token+=interval)
		{
			word=doc.getWord(token);
			oldTopic=doc.getTopicAssign(token);
			if (topics.get(oldTopic).totalTokens==0) continue;
			
			//un-assign old topic
			doc.topicCounts[oldTopic]--;
			topics.get(oldTopic).totalTokens--;
			topics.get(oldTopic).vocabCounts[word]--;
			
			//sample a new topic
			for (int topic=0; topic<param.numTopics; topic++)
			{
				topicScores[topic]=topicUpdating(docIdx, topic, word);
			}
			
			newTopic=MathUtil.selectDiscrete(topicScores);
			
			//assign the new topic
			doc.setTopicAssign(token, newTopic);
			doc.topicCounts[newTopic]++;
			topics.get(newTopic).totalTokens++;
			topics.get(newTopic).vocabCounts[word]++;
		}
	}
	
	public double topicUpdating(int docIdx, int topic, int vocab)
	{
		if (type==TRAIN)
		{
			return (alpha[topic]+corpus.get(docIdx).topicCounts[topic])*
					(param.beta+topics.get(topic).vocabCounts[vocab])/
					(param.beta*param.numVocab+topics.get(topic).totalTokens);
		}
		return (alpha[topic]+corpus.get(docIdx).topicCounts[topic])*phi[topic][vocab];
	}
	
	public void addResults(LDAResult result)
	{
		result.add(LDAResult.LOGLIKELIHOOD, logLikelihood);
		result.add(LDAResult.PERPLEXITY, perplexity);
	}
	
	public void computeLogLikelihood()
	{
		computeTheta();
		if (type==TRAIN)
		{
			computePhi();
		}
		
		int word;
		double sum;
		logLikelihood=0.0;
		for (int doc=0; doc<numDocs; doc++)
		{
			int startPos=getStartPos();
			int interval=getSampleInterval();
			for (int token=startPos; token<corpus.get(doc).docLength(); token+=interval)
			{
				word=corpus.get(doc).getWord(token);
				sum=0.0;
				for (int topic=0; topic<param.numTopics; topic++)
				{
					sum+=theta[doc][topic]*phi[topic][word];
				}
				logLikelihood+=Math.log(sum);
			}
		}
	}
	
	public void computeTheta()
	{
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				theta[doc][topic]=(alpha[topic]+corpus.get(doc).topicCounts[topic])/
						(param.alphaSum+getSampleSize(corpus.get(doc).docLength()));
			}
		}
	}
	
	public void computePhi()
	{
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				phi[topic][vocab]=(param.beta+topics.get(topic).vocabCounts[vocab])/
						(param.beta*param.numVocab+topics.get(topic).totalTokens);
			}
		}
	}
	
	public void getNumTestWords()
	{
		if (type==TRAIN)
		{
			numTestWords=numWords;
		}
		else
		{
			numTestWords=0;
			for (LDADoc doc : corpus)
			{
				numTestWords+=doc.docLength()/2;
			}
		}
	}
	
	public int getStartPos()
	{
		return (type==TRAIN ? 0 : 1);
	}
	
	public int getSampleSize(int docLength)
	{
		return (type==TRAIN ? docLength : (docLength+1)/2);
	}
	
	public int getSampleInterval()
	{
		return (type==TRAIN ? 1 : 2);
	}
	
	public String topWords(int topic, int numTopWords)
	{
		String result="Topic "+topic+":";
		LDAWord words[]=new LDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			words[vocab]=new LDAWord(param.vocabulary.get(vocab), topics.get(topic).vocabCounts[vocab]);
		}
		
		Arrays.sort(words);
		for (int i=0; i<numTopWords; i++)
		{
			result+="   "+words[i];
		}
		return result;
	}
	
	public void writeResult(String resultFileName, int numTopWords) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
		for (int topic=0; topic<param.numTopics; topic++)
		{
			bw.write(topWords(topic, numTopWords));
			bw.newLine();
		}
		bw.close();
	}
	
	public void writeTheta(String thetaFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(thetaFileName));
		Util.writeMatrix(bw, theta);
		bw.close();
	}
	
	public void writeTopicCounts(String topicCountFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(topicCountFileName));
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				bw.write(corpus.get(doc).topicCounts[topic]+" ");
			}
			bw.newLine();
		}
		bw.close();
	}
	
	public void readModel(String modelFilename) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(modelFilename));
		Util.readMatrix(br, phi);
		Util.readVector(br, alpha);
		br.close();
	}
	
	public void writeModel(String modelFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(modelFileName));
		Util.writeMatrix(bw, phi);
		Util.writeVector(bw, alpha);
		bw.close();
	}
	
	public void initVariables()
	{
		corpus=new ArrayList<LDADoc>();
		topics=new ArrayList<LDATopic>();
		alpha=new double[param.numTopics];
		phi=new double[param.numTopics][param.numVocab];
	}
	
	static
	{
		randoms=new Randoms();
	}
	
	//initialize an LDA instance for training
	public LDA(LDAParam parameters)
	{
		this.type=TRAIN;
		this.param=parameters;
		initVariables();
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			alpha[topic]=param.alphaSum/param.numTopics;
		}
	}
	
	//initialize an LDA instance from a trained LDA instance
	public LDA(LDA LDATrain, LDAParam parameters)
	{
		this.type=TEST;
		this.param=parameters;
		initVariables();
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			alpha[topic]=LDATrain.alpha[topic];
		}
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				phi[topic][vocab]=LDATrain.phi[topic][vocab];
			}
		}
	}
	
	//initialize an LDA instance from a trained LDA model in file
	public LDA(String modelFileName, LDAParam parameters) throws Exception
	{
		this.type=TEST;
		this.param=parameters;
		initVariables();
		readModel(modelFileName);
	}
	
	public static void main(String args[]) throws Exception
	{	
		String seg[]=Thread.currentThread().getStackTrace()[1].getClassName().split("\\.");
		String modelName=seg[seg.length-1];
		LDAParam parameters=new LDAParam(LDAConfig.vocabFileName);
		LDAResult trainResults=new LDAResult();
		LDAResult testResults=new LDAResult();
		
		LDA LDATrain=new LDA(parameters);
		LDATrain.readCorpus(LDAConfig.trainCorpusFileName);
		LDATrain.sample(LDAConfig.numTrainIters);
		LDATrain.addResults(trainResults);
		if (LDAConfig.SLModel)
		{
			LDATrain.writeModel(LDAConfig.getModelFileName(modelName));
		}
		
		LDA LDATest=(LDAConfig.SLModel?
				new LDA(LDAConfig.getModelFileName(modelName), parameters):
				new LDA(LDATrain, parameters));
		LDATest.readCorpus(LDAConfig.testCorpusFileName);
		LDATest.sample(LDAConfig.numTestIters);
		LDATest.addResults(testResults);
		
		trainResults.printResults(modelName+" Training Perplexity:", LDAResult.PERPLEXITY);
		testResults.printResults(modelName+" Test Perplexity:", LDAResult.PERPLEXITY);
	}
}
