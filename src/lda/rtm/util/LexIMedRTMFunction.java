package lda.rtm.util;

import lda.rtm.LexIMedRTM;
import utility.MathUtil;
import cc.mallet.optimize.Optimizable.ByGradientValue;

public class LexIMedRTMFunction implements ByGradientValue
{
	public double parameters[];
	public LexIMedRTM rtm;
	
	public LexIMedRTMFunction(LexIMedRTM RTMInst)
	{
		this.rtm=RTMInst;
		parameters=new double[rtm.param.numTopics+rtm.param.numVocab];
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			parameters[topic]=rtm.eta[topic];
		}
		for (int vocab=0; vocab<rtm.param.numVocab; vocab++)
		{
			parameters[vocab+rtm.param.numTopics]=rtm.tau[vocab];
		}
	}
	
	public double getValue()
	{
		double value=0.0,weight;
		for (int doc=0; doc<rtm.numDocs; doc++)
		{
			for (int d : rtm.trainEdgeWeights.get(doc).keySet())
			{
				weight=computeWeight(doc, d);
				value-=Math.log(1.0+Math.exp(-weight));
				if (rtm.trainEdgeWeights.get(doc).get(d)==0)
				{
					value-=weight;
				}
			}
		}
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			value-=MathUtil.sqr(parameters[topic]/rtm.param.nu)/2.0;
		}
		for (int vocab=0; vocab<rtm.param.numVocab; vocab++)
		{
			value-=MathUtil.sqr(parameters[vocab+rtm.param.numTopics]/rtm.param.nu)/2.0;
		}
		return value;
	}
	
	public void getValueGradient(double gradient[])
	{
		for (int i=0; i<gradient.length; i++)
		{
			gradient[i]=0.0;
		}
		for (int doc=0; doc<rtm.numDocs; doc++)
		{
			for (int d : rtm.trainEdgeWeights.get(doc).keySet())
			{
				double weight=computeWeight(doc, d);
				double commonTerm=Math.exp(-weight)/(1.0+Math.exp(-weight));
				for (int topic=0; topic<rtm.param.numTopics; topic++)
				{
					gradient[topic]+=commonTerm*rtm.corpus.get(doc).topicCounts[topic]/
							rtm.corpus.get(doc).docLength()*rtm.corpus.get(d).topicCounts[topic]/
							rtm.corpus.get(d).docLength();
				}
				for (int token : rtm.wordCounts.get(doc).keySet())
				{
					if (rtm.wordCounts.get(d).containsKey(token))
					{
						gradient[token+rtm.param.numTopics]+=commonTerm*rtm.wordCounts.get(doc).get(token)/
								rtm.corpus.get(doc).docLength()*rtm.wordCounts.get(d).get(token)/
								rtm.corpus.get(d).docLength();
					}
				}
				
				if (rtm.trainEdgeWeights.get(doc).get(d)==0)
				{
					for (int topic=0; topic<rtm.param.numTopics; topic++)
					{
						gradient[topic]-=1.0*rtm.corpus.get(doc).topicCounts[topic]/rtm.corpus.get(doc).docLength()*
								rtm.corpus.get(d).topicCounts[topic]/rtm.corpus.get(d).docLength();
					}
					for (int token : rtm.wordCounts.get(doc).keySet())
					{
						if (rtm.wordCounts.get(d).containsKey(token))
						{
							gradient[token+rtm.param.numTopics]-=1.0*rtm.wordCounts.get(doc).get(token)/rtm.corpus.get(doc).docLength()*
									rtm.wordCounts.get(d).get(token)/rtm.corpus.get(d).docLength();
						}
					}
				}
			}
		}
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			gradient[topic]-=parameters[topic]/MathUtil.sqr(rtm.param.nu);
		}
		for (int vocab=0; vocab<rtm.param.numVocab; vocab++)
		{
			gradient[vocab+rtm.param.numTopics]-=parameters[vocab+rtm.param.numTopics]/MathUtil.sqr(rtm.param.nu);
		}
	}
	
	public double computeWeight(int doc1, int doc2)
	{
		double weight=0.0;
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			weight+=parameters[topic]*rtm.corpus.get(doc1).topicCounts[topic]/
					rtm.corpus.get(doc1).docLength()*rtm.corpus.get(doc2).topicCounts[topic]/
					rtm.corpus.get(doc2).docLength();
		}
		for (int token : rtm.wordCounts.get(doc1).keySet())
		{
			if (rtm.wordCounts.get(doc2).containsKey(token))
			{
				weight+=parameters[token+rtm.param.numTopics]*rtm.wordCounts.get(doc1).get(token)/
						rtm.corpus.get(doc1).docLength()*rtm.wordCounts.get(doc2).get(token)/
						rtm.corpus.get(doc2).docLength();
			}
		}
		return weight;
	}
	
	public double computeEmptyWeight()
	{
		double weight=0.0;
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			weight+=parameters[topic]/MathUtil.sqr(rtm.param.numTopics);
		}
		for (int vocab=0; vocab<rtm.param.numVocab; vocab++)
		{
			weight+=parameters[vocab+rtm.param.numTopics]/MathUtil.sqr(rtm.param.numVocab);
		}
		return weight;
	}
	
	public int getNumParameters()
	{
		return parameters.length;
	}
	
	public double getParameter(int i)
	{
		return parameters[i];
	}
	
	public void getParameters(double buffer[])
	{
		for (int i=0; i<parameters.length; i++)
		{
			buffer[i]=parameters[i];
		}
	}
	
	public void setParameter(int i, double r)
	{
		parameters[i]=r;
	}
	
	public void setParameters(double newParameters[])
	{
		for (int i=0; i<parameters.length; i++)
		{
			parameters[i]=newParameters[i];
		}
	}
}
