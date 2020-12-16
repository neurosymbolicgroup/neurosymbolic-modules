package org.genesys.utils;

import com.google.gson.Gson;
import com.microsoft.z3.BoolExpr;
import com.microsoft.z3.Context;
import com.microsoft.z3.Solver;
import com.microsoft.z3.Status;
import org.genesys.models.Component;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

public class EquivGenerator {
    public static void main(String[] args) throws FileNotFoundException {
        Gson gson = new Gson();
        Z3Utils z3 = Z3Utils.getInstance();
        Solver solver_core = z3.getSolverCore();
        HashMap<String, Component> components_ = new HashMap<>();
        String specLoc = "./specs/DeepCoder";
        File[] files = new File(specLoc).listFiles();
        Set<String> constraints = new HashSet<>();
        Set<String> types = new HashSet<>();
        for (File file : files) {
            assert file.isFile() : file;
            String json = file.getAbsolutePath();
            Component comp = gson.fromJson(new FileReader(json), Component.class);
            components_.put(comp.getName(), comp);
            constraints.addAll(comp.getConstraint());
            types.add(comp.getType());
        }

        System.out.println(constraints.size());

        for (String key : constraints) {
            BoolExpr core = z3.convertStrToExpr2(key);
            for (String type : types) {
                List<String> eq_vec = new ArrayList<>();

                for (Component comp : components_.values()) {
                    if (!comp.getType().equals(type)) continue;
                    solver_core.push();
                    for (String cst : comp.getConstraint()) {
                        BoolExpr comCst = z3.convertStrToExpr2(cst);
                        solver_core.add(comCst);
                    }
                    solver_core.add(z3.getCoreCtx().mkNot(core));
                    boolean flag = (solver_core.check() == Status.SATISFIABLE);
                    solver_core.pop();
                    if (!flag && !eq_vec.contains(comp.getId())) {
                        eq_vec.add(comp.getName());
                    }
                }

                System.out.println(key + " type:" + type + "=====>" + eq_vec);
            }
        }

    }
}
