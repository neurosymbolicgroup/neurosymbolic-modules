package org.genesys.clients;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.internal.LinkedTreeMap;
import krangl.DataFrame;
import krangl.Extensions;
import krangl.SimpleDataFrameKt;
import org.genesys.decide.Decider;
import org.genesys.decide.FileDecider;
import org.genesys.decide.FirstDecider;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.interpreter.MorpheusInterpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.language.Grammar;
import org.genesys.language.MorpheusGrammar;
import org.genesys.ml.DeepCoderPythonDecider;
import org.genesys.ml.MorpheusNGramDecider;
import org.genesys.models.Example;
import org.genesys.models.Problem;
import org.genesys.synthesis.*;
import org.genesys.utils.ProblemDeserializer;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Yu on 9/3/17.
 */
public class MorpheusMain {

    public static void main(String[] args) throws FileNotFoundException {
        String json = "./problem/Morpheus/p4.json";
        String specLoc = "./specs/Morpheus";
        assert (args.length == 6);
        if (args.length != 0) json = args[0];
        specLoc = args[5];
        System.out.println("args:" + Arrays.asList(args));
//        Gson gson = new Gson();

        GsonBuilder gsonBuilder = new GsonBuilder();
        gsonBuilder.registerTypeAdapter(Problem.class, new ProblemDeserializer());
        Gson gson = gsonBuilder.create();
        Problem problem = gson.fromJson(new FileReader(json), Problem.class);
        System.out.println("Run Morpheus main..." + problem);

        Problem tableProblem = new Problem();
        List<Example> tgtExamples = new ArrayList<>();
        for(Example org : problem.getExamples()) {
            Example tgt = new Example();
            List inputTgt = new ArrayList();
            for(Object o : org.getInput()) {
                //Get the input
                LinkedTreeMap out = (LinkedTreeMap)o;
                List<String> inHeader = (List) out.get("header");
                List<String> inContent = (List) out.get("content");
                String[] arrIn = inHeader.toArray(new String[inHeader.size()]);
                DataFrame inDf = SimpleDataFrameKt.dataFrameOf(arrIn).invoke(inContent.toArray());
                System.out.println("INPUT===================");
                Extensions.print(inDf);
                inputTgt.add(inDf);
            }
            tgt.setInput(inputTgt);
            //Get the output
            LinkedTreeMap out = (LinkedTreeMap)org.getOutput();
            List<String> outHeader = (List) out.get("header");
            System.out.println("header:" + outHeader);
            List<String> outContent = (List) out.get("content");
            System.out.println("content:" + outContent);
            String[] arr = outHeader.toArray(new String[outHeader.size()]);
            DataFrame outDf = SimpleDataFrameKt.dataFrameOf(arr).invoke(outContent.toArray());
            tgt.setOutput(outDf);
            tgtExamples.add(tgt);
            System.out.println("OUTPUT=================");
            Extensions.print(outDf);
        }
        tableProblem.setExamples(tgtExamples);

        MorpheusGrammar grammar = new MorpheusGrammar(tableProblem);
        /* Load component specs. */
        Checker checker = new MorpheusChecker(specLoc, grammar);
        MorpheusInterpreter interpreter = new MorpheusInterpreter();
        // init constants in Morpheus.
        interpreter.initMorpheusConstants(grammar.getInitProductions());
        Decider decider = new FirstDecider();
        //Decider decider = new FileDecider("./sketches/sketches-size3.txt");

        boolean useStat;
//        NeoSynthesizer synth;
        MorpheusSynthesizer synth;

        useStat = Boolean.valueOf(args[3]);
        if (useStat)
            decider = new MorpheusNGramDecider();

        int depth = Integer.valueOf(args[1]);
        boolean learning = Boolean.valueOf(args[2]);

        if (!args[4].equals("") && args[3].equals("true"))
            decider = new FileDecider(args[4]);

        System.out.println("##Decider:" + decider);
        System.out.println("##Spec:" + specLoc);
        System.out.println("##Learning:" + learning);
        synth = new MorpheusSynthesizer(grammar, tableProblem, checker, interpreter, depth, specLoc, learning, decider);

        synth.synthesize();
    }
}
