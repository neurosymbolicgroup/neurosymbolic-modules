package org.genesys.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class YFeaturizer {
	public final List<String> functions = new ArrayList<String>();
	public final Map<String,Integer> functionIndices = new HashMap<String,Integer>();
	
	public YFeaturizer(List<String> functions) {
		this.functions.addAll(functions);
		for(int i=0; i<this.functions.size(); i++) {
			this.functionIndices.put(this.functions.get(i), i);
		}
	}
	
	public List<Integer> getFeatures(String function) {
		return featurize(function, this.functions);
	}
	
	private static List<Integer> featurize(String curFunction, List<String> functions) {
		List<Integer> features = new ArrayList<Integer>();
		for(String function : functions) {
			features.add(function.equals(curFunction) ? 1 : 0);
		}
		return features;
	}
}
