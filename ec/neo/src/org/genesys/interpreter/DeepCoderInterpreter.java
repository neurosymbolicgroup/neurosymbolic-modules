package org.genesys.interpreter;

import org.genesys.interpreter.deepcode.*;
import org.genesys.language.Production;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.*;

import java.util.*;

/**
 * interpreter for L2 tool. Can be used in Deepcoder
 * Created by yufeng on 5/31/17.
 */
public class DeepCoderInterpreter extends BaseInterpreter {

    public DeepCoderInterpreter() {
        executors.put("root", (objects, input) -> {
            Object obj = objects.get(0);
            if (obj instanceof Unop)
                return new Maybe<>(((Unop) objects.get(0)).apply(input));
            else
                return new Maybe<>(obj);
        });
        executors.put("input0", (objects, input) -> new Maybe<>(((List) input).get(0)));
        executors.put("input1", (objects, input) -> new Maybe<>(((List) input).get(1)));

        executors.put("-3", (objects, input) -> new Maybe<>(-3));
        executors.put("-2", (objects, input) -> new Maybe<>(-2));
        executors.put("-1", (objects, input) -> new Maybe<>(-1));
        executors.put("0", (objects, input) -> new Maybe<>(0));
        executors.put("1", (objects, input) -> new Maybe<>(1));
        executors.put("2", (objects, input) -> new Maybe<>(2));
        executors.put("3", (objects, input) -> new Maybe<>(3));

        executors.put("INC", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("+"), 1)));
        executors.put("DEC", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("-"), 1)));
        executors.put("SHL", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("*"), 2)));
        executors.put("MUL3", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("*"), 3)));
        executors.put("MUL4", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("*"), 4)));

        executors.put("SHR", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("/"), 2)));
        executors.put("DIV3", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("/"), 3)));
        executors.put("DIV4", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("/"), 4)));
        executors.put("SQR", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("**"), 2)));
        executors.put("doNEG", (objects, input) -> new Maybe<>(new NumUnop(new PrimitiveBinop("*"), -1)));

        executors.put("MAXIMUM", (objects, input) -> new Maybe<>(new MaximumUnop().apply(objects.get(0))));
        executors.put("MINIMUM", (objects, input) -> new Maybe<>(new MinimumUnop().apply(objects.get(0))));
        executors.put("SUM", (objects, input) -> new Maybe<>(new SumUnop().apply(objects.get(0))));
        executors.put("LAST", (objects, input) -> new Maybe<>(new LastUnop().apply(objects.get(0))));
        executors.put("HEAD", (objects, input) -> new Maybe<>(new HeadUnop().apply(objects.get(0))));
        executors.put("DROP", (objects, input) -> new Maybe<>(new DropUnop().apply(objects)));
        executors.put("ACCESS", (objects, input) -> new Maybe<>(new AccessUnop().apply(objects)));

        executors.put("SORT", (objects, input) -> new Maybe<>(new SortUnop().apply(objects.get(0))));
        executors.put("REVERSE", (objects, input) -> new Maybe<>(new ReverseUnop().apply(objects.get(0))));

        executors.put("MAP", (objects, input) ->
                new Maybe<>(new MapLList((Unop) objects.get(1)).apply(objects.get(0)))
        );

        executors.put("MAP-MUL", (objects, input) ->
                new Maybe<>(new MapLList((Unop)new PrimitiveUnop("*", objects.get(1))).apply(objects.get(0)))
        );
        executors.put("MAP-DIV", (objects, input) ->
                new Maybe<>(new MapLList((Unop)new PrimitiveUnop("/", objects.get(1))).apply(objects.get(0)))
        );
        executors.put("MAP-PLUS", (objects, input) ->
                new Maybe<>(new MapLList((Unop)new PrimitiveUnop("+", objects.get(1))).apply(objects.get(0)))
        );
        executors.put("MAP-POW", (objects, input) ->
                new Maybe<>(new MapLList((Unop)new PrimitiveUnop("^", objects.get(1))).apply(objects.get(0)))
        );

//        executors.put("FILTER", (objects, input) ->
//                new Maybe<>(new FilterLList((Unop) objects.get(1)).apply(objects.get(0)))
//        );

        executors.put("FILTER", (objects, input) -> {
                return new Maybe<>(new FilterLList((Binop) objects.get(1),(Integer)objects.get(2)).apply(objects.get(0)));
                }
        );

        executors.put("COUNT", (objects, input) -> {
                    return new Maybe<>(new CountList((Binop) objects.get(1),(Integer)objects.get(2)).apply(objects.get(0)));
                }
        );

//        executors.put("COUNT", (objects, input) ->
//                new Maybe<>(new CountList((Unop) objects.get(1)).apply(objects.get(0))));
//
//        executors.put("COUNT", (objects, input) ->
//                new Maybe<>(new CountList((Unop) objects.get(0)).apply(new PrimitiveUnop((String)objects.get(1),objects.get(2)))));

        executors.put("ZIPWITH", (objects, input) -> {
            assert objects.size() == 3 : objects;
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            return new Maybe<>(new ZipWith((Binop) objects.get(2)).apply(args));
        });

        executors.put("ZIPWITH-PLUS", (objects, input) -> {
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            Binop bop = new PrimitiveBinop("+");
            return new Maybe<>(new ZipWith((Binop) bop).apply(args));
        });

        executors.put("ZIPWITH-MINUS", (objects, input) -> {
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            Binop bop = new PrimitiveBinop("-");
            return new Maybe<>(new ZipWith((Binop) bop).apply(args));
        });

        executors.put("ZIPWITH-MUL", (objects, input) -> {
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            Binop bop = new PrimitiveBinop("*");
            return new Maybe<>(new ZipWith((Binop) bop).apply(args));
        });

        executors.put("ZIPWITH-MIN", (objects, input) -> {
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            Binop bop = new MinBinop();
            return new Maybe<>(new ZipWith((Binop) bop).apply(args));
        });

        executors.put("ZIPWITH-MAX", (objects, input) -> {
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            Binop bop = new MaxBinop();
            return new Maybe<>(new ZipWith((Binop) bop).apply(args));
        });


        executors.put("SCANL1", (objects, input) -> {
            assert objects.size() == 2 : objects;
            return new Maybe<>(new Scanl((Binop) objects.get(1)).apply(objects.get(0)));
        });

        executors.put("SCANL1-PLUS", (objects, input) -> {
            Binop bop = new PrimitiveBinop("+");
            return new Maybe<>(new Scanl(bop).apply(objects.get(0)));
        });

        executors.put("SCANL1-MINUS", (objects, input) -> {
            Binop bop = new PrimitiveBinop("-");
            return new Maybe<>(new Scanl(bop).apply(objects.get(0)));
        });

        executors.put("SCANL1-MUL", (objects, input) -> {
            Binop bop = new PrimitiveBinop("*");
            return new Maybe<>(new Scanl(bop).apply(objects.get(0)));
        });

        executors.put("SCANL1-MIN", (objects, input) -> {
            Binop bop = new MinBinop();
            return new Maybe<>(new Scanl(bop).apply(objects.get(0)));
        });

        executors.put("SCANL1-MAX", (objects, input) -> {
            Binop bop = new MaxBinop();
            return new Maybe<>(new Scanl(bop).apply(objects.get(0)));
        });


        executors.put("TAKE", (objects, input) -> {
            assert objects.size() == 2 : objects;
            List args = new ArrayList();
            args.add(objects.get(0));
            args.add(objects.get(1));
            return new Maybe<>(new TakeUnop().apply(args));
        });


        executors.put("+", (objects, input) -> new Maybe<>(new PrimitiveBinop("+")));
        executors.put("*", (objects, input) -> new Maybe<>(new PrimitiveBinop("*")));
        executors.put("-", (objects, input) -> new Maybe<>(new PrimitiveBinop("-")));

        executors.put("MIN", (objects, input) -> new Maybe<>(new MinBinop()));
        executors.put("MAX", (objects, input) -> new Maybe<>(new MaxBinop()));

        executors.put("l(a,b).(< a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("<")));
        executors.put("l(a,b).(> a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop(">")));
        executors.put("l(a,b).(== a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("==")));
        executors.put("l(a,b).(!= a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("!=")));
        executors.put("l(a,b).(%= a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("%=")));
        executors.put("l(a,b).(%!= a b)", (objects, input) -> new Maybe<>(new PrimitiveBinop("%!=")));

        executors.put("isPOS", (objects, input) -> new Maybe<>(new PrimitiveUnop(">", 0)));
        executors.put("isNEG", (objects, input) -> new Maybe<>(new PrimitiveUnop("<", 0)));
        executors.put("isODD", (objects, input) -> new Maybe<>(new PrimitiveUnop("%!=2", 0)));
        executors.put("isEVEN", (objects, input) -> new Maybe<>(new PrimitiveUnop("%=2", 0)));
    }

    @Override
    public Set<String> getExeKeys() {
        return this.executors.keySet();
    }
}
