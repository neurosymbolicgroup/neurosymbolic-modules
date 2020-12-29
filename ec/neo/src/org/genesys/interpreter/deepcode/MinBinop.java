package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Binop;

/**
 * Created by yufeng on 5/31/17.
 */
public class MinBinop implements Binop {

    public Object apply(Object first, Object second) {
        int i1 = (int) first;
        int i2 = (int) second;
        return i1 < i2 ? i1 : i2;
    }

    @Override
    public String toString() {
        return "MIN";
    }
}
