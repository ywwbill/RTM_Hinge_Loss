package lda.util;

import java.util.ArrayList;

public class LDADoc
{	
	private ArrayList<Integer> tokens;
	private ArrayList<Integer> topicAssign;
	
	public int topicCounts[];
	
	public LDADoc(int numTopics, int numVocab)
	{
		this("", numTopics, numVocab);
	}
	
	public LDADoc(String document, int numTopics, int numVocab)
	{
		tokens=new ArrayList<Integer>();
		topicAssign=new ArrayList<Integer>();
		topicCounts=new int[numTopics];
		
		String seg[]=document.split(" "),segseg[];
		for (int i=1; i<seg.length; i++)
		{
			if (seg[i].length()==0) continue;
			segseg=seg[i].split(":");
			int word=Integer.valueOf(segseg[0]);
			int count=Integer.valueOf(segseg[1]);
			assert(word>=0 && word<numVocab);
			for (int j=0; j<count; j++)
			{
				tokens.add(word);
				topicAssign.add(-1);
			}
		}
	}
	
	public int docLength()
	{
		return tokens.size();
	}
	
	public int getTopicAssign(int pos)
	{
		return topicAssign.get(pos);
	}
	
	public void setTopicAssign(int pos, int topic)
	{
		topicAssign.set(pos, topic);
	}
	
	public void addTopicAssign(int topic)
	{
		topicAssign.add(topic);
	}
	
	public void addWord(int word)
	{
		tokens.add(word);
	}
	
	public int getWord(int pos)
	{
		return tokens.get(pos);
	}
}
