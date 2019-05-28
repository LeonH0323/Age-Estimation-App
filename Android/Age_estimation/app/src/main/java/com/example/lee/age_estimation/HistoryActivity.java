package com.example.lee.age_estimation;

import android.app.Dialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.SimpleAdapter;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import butterknife.Bind;

public class HistoryActivity extends AppCompatActivity {

//    @Bind(R.id.list)
    public ListView listView;
    String[] pics;
    String[] ages;
    String[] times;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);

        listView = findViewById(R.id.list);
        ActionBar actionBar = getSupportActionBar();
        actionBar.setTitle("我的估计历史");
        actionBar.setDisplayHomeAsUpEnabled(true);
        actionBar.setDisplayShowHomeEnabled(false);
//        actionBar.setHomeAsUpIndicator(R.drawable.icon_bcak_1);

        init();
    }

    private void init(){

//        showPic = new Dialog(this,R.style.edit_AlertDialog_style);
//        showPic.setContentView();

        final List<Map<String,Object>> data = new ArrayList<>();
        Intent intent = getIntent();
//        byte[] bytes = intent.getByteArrayExtra("pic");
//        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes,0,bytes.length);
//        Uri uri = intent.getData();
        pics = intent.getStringArrayExtra("pic");
        ages = intent.getStringArrayExtra("age");
        times = intent.getStringArrayExtra("time");
        for (int i = 0; i < pics.length; i++) {
            Map<String,Object> item = new HashMap<>();
            Uri uri = Uri.parse(pics[i]);
            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
            item.put("image",bitmap);
            item.put("age",ages[i]);
            item.put("time",times[i]);
            Log.d("lihang", "init: "+times[i]);
            data.add(item);
        }


        SimpleAdapter simpleAdapter = new SimpleAdapter(this,
                data,
                R.layout.activity_history_item,
                new String[]{"image", "age", "time"},
                new int[]{R.id.imageView1, R.id.textView1, R.id.textView_time});
        simpleAdapter.setViewBinder(new SimpleAdapter.ViewBinder() {
            @Override
            public boolean setViewValue(View view, Object data, String textRepresentation) {
                if (view instanceof ImageView && data instanceof Bitmap) {
                    ImageView imageView = (ImageView)view;
                    Bitmap bitmap1 = (Bitmap) data;
                    imageView.setImageBitmap(bitmap1);
                    return true;
                }
                return false;
            }
        });
        listView.setAdapter(simpleAdapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Dialog dialog = new Dialog(HistoryActivity.this,R.style.edit_AlertDialog_style);
                dialog.setContentView(R.layout.acticity_showpic);
                ImageView imageView = dialog.findViewById(R.id.start_img);
                Uri uri = Uri.parse(pics[position]);
                Bitmap bitmap = null;
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(HistoryActivity.this.getContentResolver(),uri);
                } catch (IOException e) {
                    e.printStackTrace();
                }
//                imageView.setBackgroundResource(R.drawable.bg_1);
                imageView.setImageBitmap(bitmap);
                dialog.setCanceledOnTouchOutside(true);
                dialog.show();
            }
        });

    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()){
            default:
                Log.d("lihang", "onOptionsItemSelected: ？？");
                this.finish();
                break;
        }
        return super.onOptionsItemSelected(item);
    }


}
