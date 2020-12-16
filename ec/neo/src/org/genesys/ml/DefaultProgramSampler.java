package org.genesys.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.genesys.language.Grammar;
import org.genesys.language.Production;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.InitType;

public class DefaultProgramSampler<T> implements Sampler<Node> {
	private final Grammar<T> grammar;
	private final Random random;
	private final DefaultProgramSamplerParameters parameters;
	
	private final Map<T,Pair<Node,Integer>> defaultNodes = new HashMap<T,Pair<Node,Integer>>();
	
	// The depth determines how deep to build the default nodes. Note that
	// passing in depth < 0 builds the entire tree. For infinite grammars,
	// this will cause nontermination.
	public DefaultProgramSampler(Grammar<T> grammar, DefaultProgramSamplerParameters parameters, Random random) {
		this.grammar = grammar;
		this.random = random;
		this.parameters = parameters;
		this.getDefault(this.grammar.start(), parameters.maxDepth);
	}
	private Pair<Node,Integer> getDefault(T symbol, int depth) {
		if(depth == 0) {
			return new Pair<Node,Integer>(null, -1);
		}
		if(this.defaultNodes.containsKey(symbol)) {
			return this.defaultNodes.get(symbol);
		}
		Production<T> minProduction = null;
		List<Node> minNodes = null;
		int minWeight = 0;
		for(Production<T> production : this.grammar.productionsFor(symbol)) {
			int weight = 0;
			List<Node> nodes = new ArrayList<Node>();
			boolean isNull = false;
			for(T input : production.inputs) {
				Pair<Node,Integer> pair = this.getDefault(input, depth-1);
				if(pair.t0 == null) {
					isNull = true;
				}
				nodes.add(pair.t0);
				weight += pair.t1;
			}
			if(isNull) {
				continue;
			}
			if(minProduction == null || weight < minWeight) {
				minProduction = production;
				minNodes = nodes;
				minWeight = weight;
			}
		}
		if(minProduction == null) {
			this.defaultNodes.put(symbol, new Pair<Node,Integer>(null, -1));
		} else {
			this.defaultNodes.put(symbol, new Pair<Node,Integer>(new Node(minProduction.function, minNodes), minWeight));
		}
		return this.defaultNodes.get(symbol);
	}
	public Node sample() {
		Node node = this.sample(this.grammar.start(), this.parameters.maxDepth);
		if(node == null) {
			throw new RuntimeException("Insufficient depth: " + this.parameters.maxDepth);
		}
		return node;
	}
	private Node sample(T symbol, int depth) {
		if(depth == 0) {
			return this.defaultNodes.containsKey(symbol) ? this.defaultNodes.get(symbol).t0 : null;
		}
		List<Production<T>> productions = this.grammar.productionsFor(symbol);
		Production<T> production = productions.get(this.random.nextInt(productions.size()));
		List<Node> children = new ArrayList<Node>();
		for(T input : production.inputs) {
			children.add(this.sample(input, depth-1));
		}
		for(Node node : children) {
			if(node == null) {
				return this.defaultNodes.get(symbol).t0;
			}
		}
		return new Node(production.function, children);
	}
}
