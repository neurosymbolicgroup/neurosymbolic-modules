package org.genesys.utils;


import org.genesys.language.Grammar;
import org.genesys.language.Production;
import org.genesys.models.Node;
import org.genesys.type.Maybe;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SamplerUtils {
    public static class RandomNodeSampler {
        private final Random random;

        public RandomNodeSampler(Random random) {
            this.random = random;
        }

        public <T> Node sample(Grammar<T> grammar, Maybe<Node> node, T symbol) {
            return sample(grammar, symbol);
        }

        private <T> Node sample(Grammar<T> grammar, T symbol) {
            List<Production<T>> productions = grammar.productionsFor(symbol);
            Production<T> production = productions.get(this.random.nextInt(productions.size()));
            List<Node> children = new ArrayList();
            for (int i = 0; i < production.inputs.length; i++) {
                children.add(sample(grammar, production.inputs[i]));
            }
            return new Node(production.function, children);
        }
    }
}
