package org.genesys.ml;

import java.util.List;

import org.genesys.models.Quad;

public interface XFeaturizer<T> {
	// (function n-gram, list values)
	public Quad<List<List<Integer>>,List<List<Integer>>,List<List<Integer>>,List<Integer>> getFeatures(List<T> input, List<T> output, List<String> ancestors);
}
