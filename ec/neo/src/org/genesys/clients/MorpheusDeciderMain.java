package org.genesys.clients;

import java.util.ArrayList;
import java.util.List;

import org.genesys.ml.MorpheusNGramDecider;

public class MorpheusDeciderMain {
	public static void main(String[] args) {
		MorpheusNGramDecider decider = new MorpheusNGramDecider();
		List<String> ancestors = new ArrayList<String>();
		List<String> functionChoices = new ArrayList<String>();
		functionChoices.add("select");
		functionChoices.add("unite");
		System.out.println(decider.decide(ancestors, functionChoices));
		ancestors.add("select");
		System.out.println(decider.decide(ancestors, functionChoices));
	}
}
