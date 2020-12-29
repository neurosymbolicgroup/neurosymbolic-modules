package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Binop;
import org.genesys.interpreter.Unop;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class Scanl implements Unop {
    private final Binop binop;

    public Scanl(Binop binop) {
        this.binop = binop;
    }

    public Object apply(Object obj) {
        assert obj != null;
        List arg = (List) obj;
        List targetList = new ArrayList();
        for (int i = 0; i < arg.size(); i++) {
            if (i == 0) {
                targetList.add(arg.get(0));
            } else {
                Object prev = targetList.get(i - 1);
                Object val = binop.apply(prev, arg.get(i));
                targetList.add(val);
            }
        }
        return targetList;
    }

    public String toString() {
        return "SCANL1";
    }
}
