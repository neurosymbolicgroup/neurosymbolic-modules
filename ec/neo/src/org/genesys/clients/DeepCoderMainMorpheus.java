package org.genesys.clients;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.genesys.decide.Decider;
import org.genesys.decide.FileDecider;
import org.genesys.decide.FirstDecider;
import org.genesys.decide.RandomDecider;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.ml.DeepCoderPythonDecider;
import org.genesys.models.Problem;
import org.genesys.synthesis.Checker;
import org.genesys.synthesis.DeepCoderChecker;
import org.genesys.synthesis.MorpheusSynthesizer;
import org.genesys.utils.DeepCodeDeserializer;
import org.genesys.utils.ProblemDeserializer;

import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by utcs on 9/11/17.
 */
public class DeepCoderMainMorpheus {

    public static void main(String[] args) throws FileNotFoundException {
        boolean useStat;
        String specLoc = "./specs/DeepCoder";
        String json = "./problem/DeepCoder/prog5.json";
        if (args.length != 0) json = args[0];
        GsonBuilder gsonBuilder = new GsonBuilder();
        gsonBuilder.registerTypeAdapter(Problem.class, new DeepCodeDeserializer());
        Gson gson = gsonBuilder.create();
        Problem dcProblem = gson.fromJson(new FileReader(json), Problem.class);
        System.out.println("Run DeepCoder main..." + dcProblem);

        DeepCoderGrammar grammar = new DeepCoderGrammar(dcProblem);
        /* Load component specs. */
        Checker checker = new DeepCoderChecker(specLoc);
        //Checker checker = new DummyChecker();
        Interpreter interpreter = new DeepCoderInterpreter();
        Decider decider = new FirstDecider();

        MorpheusSynthesizer synth;
        assert (args.length == 5);
            useStat = Boolean.valueOf(args[3]);
            if(useStat)
                decider = new DeepCoderPythonDecider(dcProblem);

            int depth = Integer.valueOf(args[1]);
            boolean learning = Boolean.valueOf(args[2]);

            if (args.length == 5){
                if (!args[4].equals(""))
                    decider = new FileDecider(args[4]);
            }

        synth = new MorpheusSynthesizer(grammar, dcProblem, checker, interpreter, depth, specLoc, learning, decider);
        synth.synthesize();
    }
}
