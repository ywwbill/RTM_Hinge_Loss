package lda.rtm.util;

import lda.rtm.RTM;
import utility.MathUtil;
import cc.mallet.optimize.Optimizable.ByGradientValue;

public class RTMFunction implements ByGradientValue
{
	public double parameters[];
	public RTM rtm;
	
	public RTMFunction(RTM RTMInst)
	{
		this.rtm=RTMInst;
		parameters=new double[rtm.param.numTopics];
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			parameters[topic]=rtm.eta[topic];
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
				if (rtm.trainEdgeWeights.get(doc).get(d)==0)
				{
					for (int topic=0; topic<rtm.param.numTopics; topic++)
					{
						gradient[topic]-=1.0*rtm.corpus.get(doc).topicCounts[topic]/rtm.corpus.get(doc).docLength()*
								rtm.corpus.get(d).topicCounts[topic]/rtm.corpus.get(d).docLength();
					}
				}
			}
		}
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			gradient[topic]-=parameters[topic]/MathUtil.sqr(rtm.param.nu);
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
		return weight;
	}
	
	public double computeEmptyWeight()
	{
		double weight=0.0;
		for (int topic=0; topic<rtm.param.numTopics; topic++)
		{
			weight+=parameters[topic]/MathUtil.sqr(rtm.param.numTopics);
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
