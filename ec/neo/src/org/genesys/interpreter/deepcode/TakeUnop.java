package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Binop;
import org.genesys.interpreter.Unop;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class TakeUnop implements Unop {

    public TakeUnop() {
    }

    public Object apply(Object obj) {
        assert obj != null;
        List pair = (List) obj;
        if (!(pair.get(0) instanceof List) || !(pair.get(1) instanceof Integer))
            return 256;
        assert pair.size() == 2 : pair;
        assert pair.get(0) instanceof List;
        assert pair.get(1) instanceof Integer;
        List input1 = (List) pair.get(0);
        int input2 = (Integer) pair.get(1);
        if (input2 < 0) {
            return 256;
        } else if (input2 >= input1.size()) {
            return 256;
        } else {
            return input1.subList(0, input2);
        }
    }

    public String toString() {
        return "TAKE";
    }
}
