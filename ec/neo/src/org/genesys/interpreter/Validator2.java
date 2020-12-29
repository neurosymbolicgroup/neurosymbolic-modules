package org.genesys.interpreter;

import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;

import java.util.List;
import java.util.Map;

/**
 * Created by yufeng on 9/10/17.
 */
public interface Validator2 {
    Pair<Object, List<Map<Integer, List<String>>>> validate(List<Pair<Object, List<Map<Integer, List<String>>>>> objects
            , Object input, Node ast);
}
