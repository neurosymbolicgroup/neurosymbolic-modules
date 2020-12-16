package org.genesys.clients;

import com.sun.tools.javah.Gen;
import org.genesys.decide.Decider;
import org.genesys.decide.FileDecider;
import org.genesys.decide.RandomDecider;
import org.genesys.generator.Generator;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.models.Node;
import org.genesys.models.Problem;
import org.genesys.models.Trio;
import org.genesys.synthesis.Checker;
import org.genesys.synthesis.DeepCoderChecker;
import org.genesys.synthesis.GeneratorSynthesizer;
import java.io.FileNotFoundException;
import java.util.List;

/**
 * Created by ruben on 9/11/17.
 */
public class DeepCoderGenerator {

    public static void main(String[] args) throws FileNotFoundException {

        int depth = Integer.valueOf(args[0]);
        String filename = args[1];

        Generator generator = new Generator(depth, filename);
        Trio<List<List<Object>>,List<Object>,Node> program = generator.generate();

        System.out.println("inputs = " + program.t0);
        System.out.println("outputs = " + program.t1);
        System.out.println("program = " + program.t2);

    }
}
