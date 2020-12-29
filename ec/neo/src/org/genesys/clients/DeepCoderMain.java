package org.genesys.clients;

import com.google.gson.Gson;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.interpreter.L2Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.language.L2Grammar;
import org.genesys.models.Problem;
import org.genesys.synthesis.*;
import org.genesys.type.InputType;
import org.genesys.type.IntType;
import org.genesys.type.ListType;
import org.genesys.type.PairType;
import org.genesys.utils.LibUtils;

import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by yufeng on 6/4/17.
 */
public class DeepCoderMain {

    public static void main(String[] args) throws FileNotFoundException {
        String specLoc = "./specs/DeepCoder";
        String json = "./problem/DeepCoder/prog5.json";
        if (args.length != 0) json = args[0];
        Gson gson = new Gson();
        Problem dcProblem = gson.fromJson(new FileReader(json), Problem.class);
        System.out.println("Run DeepCoder main..." + dcProblem);

        DeepCoderGrammar grammar = new DeepCoderGrammar(dcProblem);
        /* Load component specs. */
        Checker checker = new DeepCoderChecker(specLoc);
        Interpreter interpreter = new DeepCoderInterpreter();
        DefaultSynthesizer synth;
        if (args.length == 2) {
            int depth = Integer.valueOf(args[1]);
            synth = new DefaultSynthesizer(grammar, dcProblem, checker, interpreter, depth);
        } else {
            synth = new DefaultSynthesizer(grammar, dcProblem, checker, interpreter);
        }
        synth.synthesize();
    }
}
