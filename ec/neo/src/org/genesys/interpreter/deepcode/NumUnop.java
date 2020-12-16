package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.PrimitiveBinop;
import org.genesys.interpreter.Unop;
import org.genesys.type.IntType;
import org.genesys.utils.LibUtils;

import java.util.Collections;
import java.util.List;

/**
 * Created by yufeng on 6/4/17.
 */
public class NumUnop implements Unop {

    private PrimitiveBinop binop;

    private Object val;

    public NumUnop(PrimitiveBinop op, Object v) {
        binop = op;
        val = v;
    }

    public Object apply(Object obj) {
        assert obj instanceof Integer;
        int v = (int) obj;
        if (binop.toString().equals("**"))
            return v * v;
        return binop.apply(v, val);
    }

    public String toString() {
        return "NumUnop_" + binop + val;
    }
}
