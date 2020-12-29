package org.genesys.interpreter;

import org.genesys.type.Maybe;

import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public interface Executor {

    Maybe<Object> execute(List<Object> objects, Object input);

}
