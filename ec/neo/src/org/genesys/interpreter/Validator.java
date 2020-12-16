package org.genesys.interpreter;

import org.genesys.models.Pair;
import org.genesys.type.Maybe;

import java.util.List;

/**
 * Created by yufeng on 9/10/17.
 */
public interface Validator {
    Pair<Boolean, Maybe<Object>> validate(List<Pair<Boolean, Maybe<Object>>> objects, Object input);
}
