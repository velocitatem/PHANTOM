import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';

export async function GET() {
  try {
    const cookieStore = await cookies();
    const supabase = createClient(cookieStore);

    const { data, error } = await supabase
      .from('tasks')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) throw error;

    return NextResponse.json({ tasks: data || [] });
  } catch (err: any) {
    console.error('tasks fetch error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const cookieStore = await cookies();
    const supabase = createClient(cookieStore);
    const body = await req.json();

    const { task_name, task_description, task_def_of_done } = body;

    if (!task_name) {
      return NextResponse.json(
        { error: 'task_name is required' },
        { status: 400 }
      );
    }

    const { data, error } = await supabase
      .from('tasks')
      .insert([{ task_name, task_description, task_def_of_done }])
      .select()
      .single();

    if (error) throw error;

    return NextResponse.json({ task: data });
  } catch (err: any) {
    console.error('task creation error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}
