package org.genesys.interpreter.L2;

import org.genesys.interpreter.Binop;
import org.genesys.interpreter.Unop;
import org.genesys.type.AbstractList;
import org.genesys.type.Cons;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 6/4/17.
 * It takes two arguments, an element and a list and returns a list with the element inserted at the first place.
 */
public class ConsV2Binop implements Binop {
    public Object apply(Object first, Object second) {
        assert first instanceof Integer;
        assert second instanceof List;
        List srcList = (List) second;
        srcList.add(0, first);
        return srcList;
    }

    public String toString() {
        return "l(a,x).(cons a x)";
    }
}