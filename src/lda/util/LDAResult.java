package lda.util;

import java.util.ArrayList;

import utility.Util;

public class LDAResult
{
	public static final int LOGLIKELIHOOD=0;
	public static final int PERPLEXITY=1;
	public static final int PLR=2;
	
	public ArrayList<Double> logLikelihood;
	public ArrayList<Double> perplexity;
	public ArrayList<Double> blockLogLikelihood;
	public ArrayList<Double> plr;
	public ArrayList<Double> error;
	
	public void add(int resultType, double result)
	{
		switch (resultType)
		{
		case LOGLIKELIHOOD: logLikelihood.add(result); break;
		case PERPLEXITY: perplexity.add(result); break;
		case PLR: plr.add(result); break;
		}
	}
	
	public void printResults(String message, int resultType)
	{
		switch (resultType)
		{
		case LOGLIKELIHOOD: printAvg(message, logLikelihood); break;
		case PERPLEXITY: printAvg(message, perplexity); break;
		case PLR: printAvg(message, plr); break;
		}
	}
	
	public static void printAvg(String message, ArrayList<Double> values)
	{
		if (values.size()==0) return;
		double avg=0.0;
		for (double value : values)
		{
			Util.println(value);
			avg+=value;
		}
		avg/=(double)values.size();
		Util.println(message+avg);
	}
	
	public LDAResult()
	{
		logLikelihood=new ArrayList<Double>();
		perplexity=new ArrayList<Double>();
		blockLogLikelihood=new ArrayList<Double>();
		plr=new ArrayList<Double>();
		error=new ArrayList<Double>();
	}
}
