package org.genesys.clients;

import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.models.Node;
import org.genesys.type.*;
import org.genesys.utils.SamplerUtils;

import java.util.*;

public class DeepCodeSampleMain {
    public static void main(String[] args) {
        DeepCoderGrammar grammar = new DeepCoderGrammar(new PairType(new ListType(new IntType()),
                new ListType(new IntType())), new ListType(new IntType()));
        Interpreter interpreter = new DeepCoderInterpreter();
        //int seed = -981645466;
        int seed = new Random().nextInt();
        Random random = new Random(seed);
        SamplerUtils.RandomNodeSampler sampler = new SamplerUtils.RandomNodeSampler(random);
        Integer[] i1 = {6, 2, 4, 7, 9};
        Integer[] i2 = {5, 3, 6, 1, 0};
        List list1 = Arrays.asList(i1);
        List list2 = Arrays.asList(i2);
        InputType in1 = new InputType(0, new ListType(new IntType()));
        InputType in2 = new InputType(1, new ListType(new IntType()));
        /* dynamically add input to grammar. */
        grammar.addInput(in1);
        grammar.addInput(in2);

        List input = new ArrayList();
        input.add(list1);
        input.add(list2);
        Node node = sampler.sample(grammar, new Maybe<>(), grammar.start());
        System.out.println("SEED: " + seed);
        System.out.println("PROGRAM: " + node.toString());
        System.out.println("INPUT: " + input);
        System.out.println("OUTPUT: " + interpreter.execute(node, input).get());
    }
}
