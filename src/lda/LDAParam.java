package lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class LDAParam
{
	//for LDA
	public double alphaSum=1.0;//alphaSum = alpha * numTopics
	public double _alphaSum=1.0;//_alphaSum = _alpha * numTopics
	public double beta=0.1;
	public int numTopics=10;
	
	public ArrayList<String> vocabulary;
	public int numVocab;
	
	//for hinge loss
	public double c=1.0;
	
	//for rtm
	public double nu=1.0;
	public int showPLRInterval=10;
	public boolean negEdge=true;//option for negative edge sampling, recommend to set it true
	public double negEdgeRatio=0.1;//the ratio of #neg-edges to #pos-edges
	public boolean directed=true;//whether the edges are directed
	
	public LDAParam(String vocabFileName) throws Exception
	{
		vocabulary=new ArrayList<String>();
		BufferedReader br=new BufferedReader(new FileReader(vocabFileName));
		String line;
		while ((line=br.readLine())!=null)
		{
			vocabulary.add(line);
		}
		br.close();
		numVocab=vocabulary.size();
	}
}
