package org.genesys.clients;

import com.google.gson.Gson;
import org.genesys.decide.Decider;
import org.genesys.decide.FirstDecider;
import org.genesys.decide.RandomDecider;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.ml.DeepCoderPythonDecider;
import org.genesys.models.Problem;
import org.genesys.synthesis.*;
import org.genesys.synthesis.NeoSynthesizer;

import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by ruben on 7/6/17.
 */
public class DeepCoderMainNeo {

    public static void main(String[] args) throws FileNotFoundException {
        boolean useStat;
        String specLoc = "./specs/DeepCoder";
        String json = "./problem/DeepCoder/prog5.json";
        if (args.length != 0) json = args[0];
        Gson gson = new Gson();
        Problem dcProblem = gson.fromJson(new FileReader(json), Problem.class);
        System.out.println("Run DeepCoder main..." + dcProblem);

        DeepCoderGrammar grammar = new DeepCoderGrammar(dcProblem);
        /* Load component specs. */
        Checker checker = new DeepCoderChecker(specLoc);
        //Checker checker = new DummyChecker();
        Interpreter interpreter = new DeepCoderInterpreter();
        Decider decider = new FirstDecider();

        NeoSynthesizer synth;
        if (args.length == 4) {
            useStat = Boolean.valueOf(args[3]);
            if(useStat)
                decider = new DeepCoderPythonDecider(dcProblem);

            int depth = Integer.valueOf(args[1]);
            boolean learning = Boolean.valueOf(args[2]);

            synth = new NeoSynthesizer(grammar, dcProblem, checker, interpreter, depth, specLoc, learning, decider);
        } else {
            synth = new NeoSynthesizer(grammar, dcProblem, checker, interpreter, specLoc, decider);
        }
        synth.synthesize();
    }
}
