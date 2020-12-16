package org.genesys.interpreter;

/**
 * Created by yufeng on 5/30/17.
 */
public interface Binop {
    Object apply(Object first, Object second);
}
