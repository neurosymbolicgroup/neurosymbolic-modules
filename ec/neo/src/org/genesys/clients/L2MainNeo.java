package org.genesys.clients;

import com.google.gson.Gson;
import org.genesys.decide.Decider;
import org.genesys.decide.FirstDecider;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.interpreter.L2V2Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.language.Grammar;
import org.genesys.language.L2V2Grammar;
import org.genesys.ml.DeepCoderPythonDecider;
import org.genesys.models.Problem;
import org.genesys.synthesis.Checker;
import org.genesys.synthesis.DeepCoderChecker;
import org.genesys.synthesis.NeoSynthesizer;

import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by ruben on 7/6/17.
 */
public class L2MainNeo {

    public static void main(String[] args) throws FileNotFoundException {
        String specLoc = "./specs/DeepCoder";
        String json = "./problem/DeepCoder/prog5.json";
        if (args.length != 0) json = args[0];
        Gson gson = new Gson();
        Problem dcProblem = gson.fromJson(new FileReader(json), Problem.class);
        System.out.println("Run DeepCoder main..." + dcProblem);

        Grammar grammar = new L2V2Grammar(dcProblem);
        /* Load component specs. */
        Checker checker = new DeepCoderChecker(specLoc);
        Interpreter interpreter = new L2V2Interpreter();
        Decider decider = new FirstDecider();

        NeoSynthesizer synth;
        assert args.length == 4;
        boolean useStat = Boolean.valueOf(args[3]);
        if (useStat)
            decider = new DeepCoderPythonDecider(dcProblem);

        int depth = Integer.valueOf(args[1]);
        boolean learning = Boolean.valueOf(args[2]);

        synth = new NeoSynthesizer(grammar, dcProblem, checker, interpreter, depth, specLoc, learning, decider);
        synth.synthesize();
    }
}
