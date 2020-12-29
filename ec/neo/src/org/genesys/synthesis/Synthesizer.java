package org.genesys.synthesis;

import org.genesys.interpreter.Interpreter;
import org.genesys.language.Grammar;
import org.genesys.models.Node;

/**
 * Created by yufeng on 5/28/17.
 */
public interface Synthesizer<T> {
    T synthesize();
}