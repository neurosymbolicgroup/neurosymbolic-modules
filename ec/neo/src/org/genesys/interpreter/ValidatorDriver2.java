package org.genesys.interpreter;

import org.genesys.models.Pair;
import org.genesys.type.Maybe;

import java.util.List;
import java.util.Map;

/**
 * Created by yufeng on 9/10/17.
 */

public interface ValidatorDriver2<T, I> {
    Pair<Object, List<Map<Integer, List<String>>>> validate(T node, I input);

    Object getPE(int key);

    void cleanPEMap();
}
