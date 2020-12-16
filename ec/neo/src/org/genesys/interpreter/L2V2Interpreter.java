package org.genesys.interpreter;

import org.genesys.interpreter.L2.ConsV2Binop;
import org.genesys.interpreter.L2.Foldr;
import org.genesys.interpreter.deepcode.*;
import org.genesys.language.Production;
import org.genesys.models.Node;
import org.genesys.type.*;

import java.util.*;

/**
 * interpreter for L2 tool. Can be used in Deepcoder
 * Created by yufeng on 5/31/17.
 */
public class L2V2Interpreter extends BaseInterpreter {

    public L2V2Interpreter() {
        executors.put("root", (objects, input) -> {
            Object obj = objects.get(0);
            if (obj instanceof Unop)
                return new Maybe<>(((Unop) objects.get(0)).apply(input));
            else
                return new Maybe<>(obj);
        });
        executors.put("input0", (objects, input) -> new Maybe<>(((List) input).get(0)));
        executors.put("input1", (objects, input) -> new Maybe<>(((List) input).get(1)));
        executors.put("true", (objects, input) -> new Maybe<>(true));
        executors.put("false", (objects, input) -> new Maybe<>(false));
        executors.put("0", (objects, input) -> new Maybe<>(0));
        executors.put("1", (objects, input) -> new Maybe<>(1));
        executors.put("-1", (objects, input) -> new Maybe<>(-1));
        executors.put("+1", (objects, input) -> new Maybe<>((int) objects.get(0) + 1));
        executors.put("*3", (objects, input) -> new Maybe<>((int) objects.get(0) * 3));
        executors.put("maximum", (objects, input) -> new Maybe<>(new MaximumUnop().apply(objects.get(0))));
        executors.put("minimum", (objects, input) -> new Maybe<>(new MinimumUnop().apply(objects.get(0))));
        executors.put("sum", (objects, input) -> new Maybe<>(new SumUnop().apply(objects.get(0))));
        executors.put("last", (objects, input) -> new Maybe<>(new LastUnop().apply(objects.get(0))));
        executors.put("head", (objects, input) -> new Maybe<>(new HeadUnop().apply(objects.get(0))));
        executors.put("sort", (objects, input) -> new Maybe<>(new SortUnop().apply(objects.get(0))));
        executors.put("reverse", (objects, input) -> new Maybe<>(new ReverseUnop().apply(objects.get(0))));
        executors.put("-", (objects, input) -> new Maybe<>(-(int) objects.get(0)));
        executors.put("map", (objects, input) ->
                new Maybe<>(new MapLList((Unop) objects.get(0)).apply(objects.get(1)))
        );
//        executors.put("filter", (objects, input) ->
//                new Maybe<>(new FilterLList((Unop) objects.get(0)).apply(objects.get(1)))
//        );
//        executors.put("count", (objects, input) ->
//                new Maybe<>(new CountList((Unop) objects.get(0)).apply(objects.get(1))));
        executors.put("zipWith", (objects, input) -> {
            assert objects.size() == 3 : objects;
            List args = new ArrayList();
            args.add(objects.get(1));
            args.add(objects.get(2));
            return new Maybe<>(new ZipWith((Binop) objects.get(0)).apply(args));
        });
        executors.put("take", (objects, input) -> {
            assert objects.size() == 2 : objects;
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            return new Maybe<>(new TakeUnop().apply(args));
        });
        executors.put("l(a,b).(+ a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("+")));
        executors.put("l(a,b).(min a b)", (objects, input) -> new Maybe<>(new MinBinop()));
        executors.put("l(a,b).(* a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("*")));
        executors.put("l(a,b).(% a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("%")));
        executors.put("l(a,b).(> a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop(">")));
        executors.put("l(a,b).(< a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("<")));
        executors.put("l(a,b).(== a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("==")));
        executors.put("l(a,b).(|| a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("||")));
        executors.put("l(a,b).(&& a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("&&")));
        executors.put("l(a).(+ a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("+", objects.get(0))));
        executors.put("l(a).(* a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("*", objects.get(0))));
        executors.put("l(a).(> a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop(">", objects.get(0))));
        executors.put("l(a).(< a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("<", objects.get(0))));
        executors.put("l(a).(== a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("==", objects.get(0))));
        executors.put("l(a).(|| a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("||", objects.get(0))));
        executors.put("l(a).(%!=2 a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("%!=2", objects.get(0))));
        executors.put("l(a).(&& a b)", (objects, input) -> new Maybe<>(new PrimitiveUnop("&&", objects.get(0))));
        executors.put("l(a).(- a)", (objects, input) -> new Maybe<>(new PrimitiveUnop("-", null)));
        executors.put("l(a).(~ a)", (objects, input) -> new Maybe<>(new PrimitiveUnop("~", null)));

//        productions.add(new Production<>(O2, "foldRight", new FunctionType(new PairType(I, O2), O2), O2, new ListType(I)));
//        productions.add(new Production<>(new FunctionType(new PairType(I, O2), O2), "l(a,x).(cons a x)"));

        executors.put("l(a,x).(cons a x)", (objects, input) -> new Maybe<>(new ConsV2Binop()));
        executors.put("foldRight", (objects, input) -> new Maybe<Object>(new Foldr((Binop) objects.get(0), objects.get(1), objects.get(2))));
    }

    @Override
    public Set<String> getExeKeys() {
        return this.executors.keySet();
    }
}
