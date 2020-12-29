package org.genesys.interpreter;

import org.genesys.models.Pair;
import org.genesys.type.Maybe;

/**
 * Created by yufeng on 9/10/17.
 */

public interface ValidatorDriver<T, I> {
    Pair<Boolean, Maybe<I>> validate(T node, I input);

    Object getPE(int key);

    void cleanPEMap();
}
