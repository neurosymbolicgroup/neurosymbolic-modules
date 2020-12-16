package org.genesys.clients;

import com.google.gson.*;
import org.genesys.interpreter.Interpreter;
import org.genesys.interpreter.L2Interpreter;
import org.genesys.language.L2Grammar;
import org.genesys.models.Problem;
import org.genesys.synthesis.Checker;
import org.genesys.synthesis.DefaultSynthesizer;
import org.genesys.synthesis.DummyChecker;
import org.genesys.synthesis.Synthesizer;
import org.genesys.type.IntType;
import org.genesys.type.ListType;

import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by yufeng on 6/4/17.
 */
public class L2Main {

    public static void main(String[] args) throws FileNotFoundException {
        String json = "./problem/L2/reverse.json";
        if (args.length != 0) json = args[0];
        Gson gson = new Gson();
        Problem l2Problem = gson.fromJson(new FileReader(json), Problem.class);
        System.out.println("Run L2 main..." + l2Problem);

        L2Grammar l2Grammar = new L2Grammar(new ListType(new IntType()), new ListType(new IntType()));
        Checker checker = new DummyChecker();
        Interpreter interpreter = new L2Interpreter();
        Synthesizer synth = new DefaultSynthesizer(l2Grammar, l2Problem, checker, interpreter);
        synth.synthesize();
    }
}
