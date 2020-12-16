package org.genesys.interpreter;

import org.genesys.type.Maybe;

import java.util.Set;


/**
 * Created by yufeng on 5/30/17.
 */
public interface Interpreter<T, I> {
    Maybe<I>    execute(T node, I input);

    Set<String> getExeKeys();
}