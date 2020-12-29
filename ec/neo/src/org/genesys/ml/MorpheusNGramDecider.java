package org.genesys.ml;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.genesys.decide.Decider;

public class MorpheusNGramDecider implements Decider {
	private static final String NO_FUNCTION = "";
	private static final String N_GRAM_FILENAME = "./model/data/morpheus_ngram_weights.txt";
	private final Map<String,Map<String,Double>> weights = new HashMap<String,Map<String,Double>>();
	
	public MorpheusNGramDecider() {
		try {
			BufferedReader br = new BufferedReader(new FileReader(N_GRAM_FILENAME));
			String line;
			while((line = br.readLine()) != null) {
				String[] tokens = line.trim().split("\\s+");
				for(int i=1; i<=2; i++) {
					if(tokens[i].equals("<s>") || tokens[i].equals("</s>")) {
						tokens[i] = NO_FUNCTION;
					}
				}
				double weight = Double.parseDouble(tokens[0]);
				if(!this.weights.containsKey(tokens[1])) {
					this.weights.put(tokens[1], new HashMap<String,Double>());
				}
				this.weights.get(tokens[1]).put(tokens[2], weight);
			}
			br.close();
		} catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public String decideSketch(List<String> trail, List<String> candidates, int child) { return decide(trail, candidates); }

	@Override
	public String decide(List<String> ancestors, List<String> functionChoices) {
		String curFunction = null;
		double curWeight = 0.0;
		String parent = ancestors.size() == 0 ? NO_FUNCTION : ancestors.get(ancestors.size()-1);
		for(String function : functionChoices) {
			double weight;
			if(!this.weights.containsKey(parent)) {
				weight = Double.MIN_VALUE;
			} else if(!this.weights.get(parent).containsKey(function)) {
				weight = Double.MIN_VALUE;
			} else {
				weight = this.weights.get(parent).get(function);
			}
			if(curFunction == null || weight > curWeight) {
				curFunction = function;
				curWeight = weight;
			}
		}
		if(curFunction == null) {
			throw new RuntimeException("No valid choices provided!");
		}
		return curFunction;
	}
}
